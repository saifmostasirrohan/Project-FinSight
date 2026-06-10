import json

import pandas as pd
import structlog

from services.database import SupabaseManager

logger = structlog.get_logger()


class ForensicAnalytics:
    @classmethod
    def get_raw_transactions(cls, session_id: str) -> list[dict]:
        """Extract unique source rows from chunk metadata for one session."""
        supabase = SupabaseManager.get_client()
        db_res = (
            supabase.table("chunks")
            .select("metadata")
            .eq("session_id", session_id)
            .execute()
        )

        transactions = []
        seen_rows = set()
        for record in db_res.data or []:
            metadata = record.get("metadata") or {}
            raw_row = metadata.get("raw_row_data")
            if not isinstance(raw_row, dict):
                continue

            row_fingerprint = json.dumps(
                raw_row,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if row_fingerprint in seen_rows:
                continue

            seen_rows.add(row_fingerprint)
            transactions.append(raw_row)

        logger.info(
            "raw_transactions_loaded_for_analysis",
            session_id=session_id,
            transaction_count=len(transactions),
        )
        return transactions

    @staticmethod
    def _transaction_frame(transactions: list[dict]) -> pd.DataFrame:
        """Normalize required ledger fields into analytics-friendly columns."""
        if not transactions:
            return pd.DataFrame()

        frame = pd.DataFrame(transactions)
        required_columns = {"amount", "date", "vendor"}
        if not required_columns.issubset(frame.columns):
            logger.warning(
                "analytics_required_columns_missing",
                missing_columns=sorted(required_columns - set(frame.columns)),
            )
            return pd.DataFrame()

        frame["amount_num"] = pd.to_numeric(
            frame["amount"].astype(str).str.replace(r"[^\d.]", "", regex=True),
            errors="coerce",
        )
        frame["date_dt"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["amount_num", "date_dt", "vendor"])
        return frame.sort_values(by="date_dt")

    @classmethod
    def detect_duplicates(cls, session_id: str) -> list[dict]:
        """Flag matching vendor/amount payments occurring within 48 hours."""
        frame = cls._transaction_frame(cls.get_raw_transactions(session_id))
        if len(frame) < 2:
            return []

        duplicates = []
        for (vendor, amount), group in frame.groupby(
            ["vendor", "amount_num"],
            dropna=False,
        ):
            if len(group) < 2:
                continue

            ordered_group = group.sort_values("date_dt").copy()
            time_deltas = ordered_group["date_dt"].diff().dt.days
            close_intervals = time_deltas.between(0, 2, inclusive="both")
            if not close_intervals.any():
                continue

            risk_score = 7.5 + (1.5 if len(ordered_group) > 2 else 0.0)
            duplicates.append(
                {
                    "vendor": vendor,
                    "amount": float(amount),
                    "currency": cls._extract_currency(ordered_group.iloc[0]["amount"]),
                    "transaction_ids": ordered_group.get(
                        "txn_id",
                        pd.Series(["UNKNOWN"] * len(ordered_group)),
                    ).tolist(),
                    "dates": ordered_group["date"].astype(str).tolist(),
                    "risk_score": min(risk_score, 10.0),
                    "details": (
                        f"Detected {len(ordered_group)} matching vendor and amount "
                        "transactions with at least one interval within 48 hours."
                    ),
                }
            )
        return duplicates

    @classmethod
    def detect_outliers(cls, session_id: str) -> list[dict]:
        """Flag high-value transactions using a robust session distribution."""
        frame = cls._transaction_frame(cls.get_raw_transactions(session_id))
        if len(frame) < 3:
            return []

        amounts = frame["amount_num"]
        first_quartile = amounts.quantile(0.25)
        third_quartile = amounts.quantile(0.75)
        interquartile_range = third_quartile - first_quartile

        if interquartile_range > 0:
            threshold = third_quartile + (1.5 * interquartile_range)
            detection_method = "the session IQR upper bound"
        else:
            median = amounts.median()
            threshold = median * 3
            detection_method = "three times the session median"

        outliers = []
        for _, row in frame[frame["amount_num"] > threshold].iterrows():
            outliers.append(
                {
                    "txn_id": row.get("txn_id", "UNKNOWN"),
                    "date": row.get("date"),
                    "vendor": row.get("vendor"),
                    "amount": row.get("amount"),
                    "risk_score": 8.5,
                    "reason": (
                        "Statistical outlier. Transaction value exceeds "
                        f"{detection_method} of {threshold:.2f}."
                    ),
                }
            )
        return outliers

    @staticmethod
    def _extract_currency(amount: object) -> str:
        tokens = str(amount).strip().split()
        return tokens[-1] if len(tokens) > 1 else "UNKNOWN"
