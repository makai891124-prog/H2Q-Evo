# Final Demo Scorecard

- generated_at_utc: `2026-03-06T15:11:48.444159+00:00`
- overall_score: `100`
- grade: `A`

## Inputs

- step1_protocol_artifact: `/Users/imymm/H2Q-Evo/reports/one_click_step1_responses_1772809750.json`
- step2_full_chain_artifact: `/Users/imymm/H2Q-Evo/reports/one_click_step2_full_1772809370.json`
- step3_trusted_chat_artifact: `/Users/imymm/H2Q-Evo/reports/trusted_local_agi_chat_session_1772809684.json`

## Step Scores

- step1_protocol: score=`100`, verdict=`pass`
- step2_full_chain: score=`100`, verdict=`pass`
- step3_trusted_chat: score=`100`, verdict=`pass`

## Key Checks

- step1 checks: `{'object_is_response': True, 'status_completed': True, 'has_output_text': True, 'has_usage': True}`
- step2 checks: `{'mode_full': True, 'ok_true': True, 'confidence_present': True, 'artifacts_present': True}`
- step3 checks: `{'trusted_ready': True, 'trust_score_gte_07': True, 'has_transcript': True, 'assistant_non_empty': True, 'contains_code_fence': True, 'has_route': True, 'latency_recorded': True}`

## Notes

- This scorecard reflects runnable integration quality, not benchmark superiority claims.
- Re-run one_click_agi_experience.sh before regenerating for fresh evidence.
