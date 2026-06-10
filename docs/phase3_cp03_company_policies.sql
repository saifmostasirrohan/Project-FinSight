CREATE TABLE IF NOT EXISTS company_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    max_allowable_amount NUMERIC,
    requires_approval BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS unique_company_policy_session_rule_idx
ON company_policies (session_id, rule_text);

INSERT INTO company_policies (
    session_id,
    rule_text,
    max_allowable_amount,
    requires_approval
)
VALUES (
    'session-anomaly-audit-101',
    'No single payment to a culinary, restaurant, or standard vendor above 20,000 BDT without supervisor approval.',
    20000.00,
    TRUE
)
ON CONFLICT (session_id, rule_text)
DO UPDATE SET
    max_allowable_amount = EXCLUDED.max_allowable_amount,
    requires_approval = EXCLUDED.requires_approval;
