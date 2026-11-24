
from resq_sense.triage import assess_posture

tests = [
    ([10, 10, 150, 50], "Lying / Horizontal", "CRITICAL"),
    ([10, 10, 100, 100], "Crouched / Partial", "WARNING"),
    ([10, 10, 50, 150], "Standing / Vertical", "AMBULATORY"),
]

for box, expected_posture, expected_priority in tests:
    posture, priority = assess_posture(box)

    print(box, "=>", posture, priority)

    assert posture == expected_posture
    assert priority == expected_priority

print("\nSUCCESS: All posture tests passed.")
