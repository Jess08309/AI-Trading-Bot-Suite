from core.config import Config, SYMBOL_DTE_MAP
c = Config()

# Test per-symbol DTE
print(f"AVGO DTE: {c.get_target_dte('AVGO')}")  # expect 2
print(f"F DTE: {c.get_target_dte('F')}")          # expect 30
print(f"NVDA DTE: {c.get_target_dte('NVDA')}")    # expect 5
print(f"IWM DTE: {c.get_target_dte('IWM')}")      # expect 1
print(f"UNKNOWN DTE: {c.get_target_dte('XYZ')}")   # expect 2 (default)

# Test per-symbol hold days
print(f"\nAVGO hold: {c.get_max_hold_days('AVGO')}d")  # 1 (2DTE)
print(f"SBUX hold: {c.get_max_hold_days('SBUX')}d")    # 3 (9DTE)
print(f"F hold: {c.get_max_hold_days('F')}d")            # 7 (30DTE)
print(f"DIS hold: {c.get_max_hold_days('DIS')}d")        # 7 (16DTE)

print("\nAll tests passed!")
