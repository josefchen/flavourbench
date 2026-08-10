-- Migration 0014 installs the stronger response-arm evidence guard.  The
-- narrower 0009 trigger cannot survive the 0013 digest backfill because it
-- compares PostgreSQL json values directly.  Migration 0016 already removes
-- this exact trigger; the legacy bridge performs that removal before 0013.
DROP TRIGGER IF EXISTS trg_response_arm_contract_immutable ON public.response_arms;
