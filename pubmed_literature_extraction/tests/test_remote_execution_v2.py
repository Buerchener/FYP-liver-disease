import unittest

from cognitive_agent.remote_execution import (
    RemoteExecutionGateway, RemoteExecutionLedger, stable_request_id,
)


class RemoteExecutionV2Tests(unittest.TestCase):
    def request_id(self):
        return stable_request_id(
            tool="claim_gate", stage="gate", round_index=1, batch=0,
            model="deepseek", schema_version="v2", input_hash="abc",
        )

    def test_cold_and_warm_share_logical_request_identity(self):
        cache = {}

        def run():
            ledger = RemoteExecutionLedger(4, 8)
            gateway = RemoteExecutionGateway(ledger, soft_timeout_s=0.00001)
            calls = []

            def lookup(factory):
                key = self.request_id()
                if key in cache:
                    return cache[key], "persistent_hit"
                calls.append(key)
                cache[key] = factory()
                return cache[key], "miss"

            result, _, record = gateway.execute(
                request_id=self.request_id(), tool="claim_gate", stage="gate",
                round_index=1, batch=0, cache_lookup=lookup,
                invoke=lambda: {"status": "OK", "attempts": 2},
                attempts_of=lambda value: value["attempts"],
                status_of=lambda value: value["status"],
            )
            return result, record, ledger, calls

        _, cold, cold_ledger, cold_calls = run()
        _, warm, warm_ledger, warm_calls = run()
        self.assertEqual(cold.request_id, warm.request_id)
        self.assertEqual(cold.logical_step, warm.logical_step)
        self.assertEqual(cold_ledger.physical_remote_attempts, 2)
        self.assertEqual(warm_ledger.physical_remote_attempts, 0)
        self.assertEqual(len(cold_calls), 1)
        self.assertEqual(warm_calls, [])

    def test_duplicate_request_is_deduplicated(self):
        ledger = RemoteExecutionLedger(2, 4)
        gateway = RemoteExecutionGateway(ledger)
        calls = []

        def lookup(factory):
            calls.append(1)
            return factory(), "miss"

        args = dict(
            request_id=self.request_id(), tool="claim_gate", stage="gate",
            round_index=1, batch=0, cache_lookup=lookup,
            invoke=lambda: {"status": "OK"}, status_of=lambda value: value["status"],
        )
        gateway.execute(**args)
        _, _, duplicate = gateway.execute(**args)
        self.assertEqual(calls, [1])
        self.assertTrue(duplicate.cache_replayed)
        self.assertEqual(ledger.logical_aux_steps, 1)
