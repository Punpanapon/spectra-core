#!/usr/bin/env python3
import argparse

def main():
    parser = argparse.ArgumentParser(description="Estimate App Runner costs")
    parser.add_argument("--vcpu", type=float, default=1, help="vCPU per instance")
    parser.add_argument("--memgb", type=float, default=2, help="GB memory per instance")
    parser.add_argument("--active-h-per-day", type=float, default=2, help="Hours processing requests per day")
    parser.add_argument("--paused-h-per-day", type=float, default=22, help="Hours paused per day")
    parser.add_argument("--credit", type=float, default=100, help="USD credits available")
    parser.add_argument("--vcpu-rate", type=float, default=0.064, help="USD per vCPU-hour")
    parser.add_argument("--mem-rate", type=float, default=0.007, help="USD per GB-hour")
    
    args = parser.parse_args()
    
    # Calculate costs
    daily_compute = args.active_h_per_day * (args.vcpu * args.vcpu_rate)
    daily_provisioned = (24 - args.paused_h_per_day) * (args.memgb * args.mem_rate)
    daily_total = daily_compute + daily_provisioned
    monthly_total = daily_total * 30
    days_covered = args.credit / daily_total if daily_total > 0 else 99999
    
    # Print results
    print("=" * 50)
    print("AWS App Runner Cost Estimate")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  vCPU: {args.vcpu}")
    print(f"  Memory: {args.memgb} GB")
    print(f"  Active hours/day: {args.active_h_per_day}")
    print(f"  Paused hours/day: {args.paused_h_per_day}")
    print(f"  Available credits: ${args.credit}")
    print()
    print(f"Daily Costs:")
    print(f"  Compute (active): ${daily_compute:.4f}")
    print(f"  Provisioned (non-paused): ${daily_provisioned:.4f}")
    print(f"  Total per day: ${daily_total:.4f}")
    print()
    print(f"Monthly estimate: ${monthly_total:.2f}")
    print(f"Credits will last: {days_covered:.1f} days")
    print("=" * 50)

if __name__ == "__main__":
    main()