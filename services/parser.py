import io
import re

import pandas as pd
import structlog
from pypdf import PdfReader

logger = structlog.get_logger()


class DocumentParser:
    @staticmethod
    def clean_text(text: str) -> str:
        """Remove whitespace clusters and normalize non-ASCII character leakage."""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        return text.strip()

    @classmethod
    def parse_pdf(cls, file_bytes: bytes) -> list[dict]:
        """Extract normalized text blocks per PDF page."""
        logger.info("initiating_pdf_text_extraction_sequence")
        results = []

        try:
            file_stream = io.BytesIO(file_bytes)
            reader = PdfReader(file_stream)
            total_pages = len(reader.pages)

            for index, page in enumerate(reader.pages):
                page_num = index + 1
                raw_content = page.extract_text()
                cleaned_content = cls.clean_text(raw_content)

                if len(cleaned_content) < 50:
                    logger.debug(
                        "skipping_low_density_pdf_page",
                        page=page_num,
                        char_count=len(cleaned_content),
                    )
                    continue

                results.append(
                    {
                        "source_location": page_num,
                        "content": cleaned_content,
                        "metadata": {
                            "char_count": len(cleaned_content),
                            "file_type": "application/pdf",
                        },
                    }
                )

            logger.info(
                "pdf_extraction_complete",
                extracted_pages=len(results),
                total_source_pages=total_pages,
            )
            return results
        except Exception as exc:
            logger.error("pdf_extraction_failed", error=str(exc))
            raise RuntimeError(
                f"PDF Structural Extraction Engine Fault: {str(exc)}"
            ) from exc

    @classmethod
    def parse_csv(cls, file_bytes: bytes) -> list[dict]:
        """Convert ledger rows into context-preserving natural language records."""
        logger.info("initiating_csv_tabular_extraction_sequence")
        results = []

        try:
            file_stream = io.BytesIO(file_bytes)
            df = pd.read_csv(file_stream, skipinitialspace=True)
            df.columns = [str(col).lower().strip() for col in df.columns]

            col_map = {
                "date": next(
                    (
                        c
                        for c in df.columns
                        if any(k in c for k in ["date", "timestamp", "time"])
                    ),
                    None,
                ),
                "amount": next(
                    (
                        c
                        for c in df.columns
                        if any(
                            k in c
                            for k in ["amount", "volume", "debit", "credit", "txn_amt"]
                        )
                    ),
                    None,
                ),
                "vendor": next(
                    (
                        c
                        for c in df.columns
                        if any(
                            k in c
                            for k in [
                                "vendor",
                                "merchant",
                                "payee",
                                "recipient",
                                "description",
                            ]
                        )
                    ),
                    None,
                ),
                "id": next(
                    (
                        c
                        for c in df.columns
                        if any(k in c for k in ["id", "hash", "ref", "txn_id"])
                    ),
                    None,
                ),
            }

            logger.debug("csv_column_mapping_resolved", resolved_map=col_map)

            for index, row in df.iterrows():
                row_num = index + 1

                date_val = (
                    str(row[col_map["date"]]).strip()
                    if col_map["date"]
                    else "UNKNOWN_DATE"
                )
                amt_val = (
                    str(row[col_map["amount"]]).strip()
                    if col_map["amount"]
                    else "0.00"
                )
                vendor_val = (
                    str(row[col_map["vendor"]]).strip()
                    if col_map["vendor"]
                    else "UNKNOWN_VENDOR"
                )
                id_val = (
                    str(row[col_map["id"]]).strip()
                    if col_map["id"]
                    else f"ROW-{row_num}"
                )

                misc_data = {}
                mapped_columns = {value for value in col_map.values() if value}
                for col in df.columns:
                    if col not in mapped_columns and pd.notna(row[col]):
                        misc_data[col] = str(row[col]).strip()

                structured_sentence = (
                    f"Financial Transaction Record ID {id_val}: On date {date_val}, "
                    f"a transaction valued at {amt_val} was executed involving vendor "
                    f"or description '{vendor_val}'."
                )
                if misc_data:
                    structured_sentence += (
                        f" Additional context metrics: {str(misc_data)}."
                    )

                results.append(
                    {
                        "source_location": row_num,
                        "content": structured_sentence,
                        "metadata": {
                            "raw_row_data": row.to_dict(),
                            "file_type": "text/csv",
                        },
                    }
                )

            logger.info("csv_extraction_complete", processed_rows=len(results))
            return results
        except Exception as exc:
            logger.error("csv_extraction_failed", error=str(exc))
            raise RuntimeError(
                f"CSV Tabular Parsing Engine Fault: {str(exc)}"
            ) from exc
