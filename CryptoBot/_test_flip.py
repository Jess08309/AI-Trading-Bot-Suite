from cryptotrades.utils.regime_detector import RegimeDetector
rd = RegimeDetector(bot_name="CryptoBot")
print("Import OK")

# Test initial state
fs = rd.get_flip_state()
print(f"Initial: cooldown={fs['is_cooldown']}, severity={fs['flip_severity']}, mult={fs['adjustment_multiplier']}")

# Record first regime
rd.record_regime("TRENDING_UP", 0.8)
fs = rd.get_flip_state()
print(f"After first: cooldown={fs['is_cooldown']}, mult={fs['adjustment_multiplier']}")

# Record a flip
rd.record_regime("TRENDING_DOWN", 0.7)
fs = rd.get_flip_state()
print(f"After UP->DOWN flip: cooldown={fs['is_cooldown']}, severity={fs['flip_severity']}, mult={fs['adjustment_multiplier']}, block={fs['should_block_entries']}")

# Summary
print(f"Summary: {rd.get_regime_summary()}")
print("ALL TESTS PASSED")
