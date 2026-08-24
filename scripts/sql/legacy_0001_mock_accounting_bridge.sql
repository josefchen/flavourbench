-- Normalize only the four content-addressed, zero-cost development fixtures
-- from the frozen 0001 backup.  Their existing endpoint receipts, amounts,
-- identities, exploratory classification, and rank=false status are unchanged.
UPDATE public.response_arms
SET cost_accounting_basis = 'mock_fixture',
    billing_reconciliation_status = 'not_applicable'
WHERE id IN (
    '18818502-ed30-4a3a-970e-18813c8d8b0d',
    '65f4ebad-591c-4268-adbc-e4f6dffb0bb0',
    '8829771a-9268-4865-a9a6-889a502a2842',
    'afa542d5-5278-417c-ba1a-cd5d23646047'
)
  AND status = 'complete'
  AND provider_slug = 'mock'
  AND actual_provider_slug = 'mock'
  AND actual_model_id LIKE 'flavourbench/mock-%'
  AND generation_id = 'mock-' || id
  AND cost_micros = 0
  AND cost_reconciled IS TRUE
  AND cost_accounting_basis = 'unrecorded'
  AND billing_reconciliation_status = 'unrecorded'
  AND EXISTS (
      SELECT 1
      FROM public.cost_events AS receipt
      WHERE receipt.arm_id = response_arms.id
        AND receipt.kind = 'actual'
        AND receipt.amount_micros = 0
        AND receipt.provider = 'mock'
        AND receipt.generation_id = response_arms.generation_id
        AND receipt.accounting_json::jsonb ->> 'reconciled' = 'true'
  );
