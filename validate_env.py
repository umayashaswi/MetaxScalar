#!/usr/bin/env python
"""
Validation script for Customer Support Environment
Run with: python validate_env.py
"""

import sys
import json
import asyncio
from pathlib import Path
from app.env import CustomerSupportEnv
from app.models import Action

async def validate_task(task_id):
    """Validate a single task"""
    print(f"\n{'='*50}")
    print(f"Validating task: {task_id}")
    print(f"{'='*50}")
    
    env = CustomerSupportEnv(task_id)
    obs = env.reset()
    
    print(f"✓ Environment reset successful")
    print(f"  - Task ID: {obs.task_id}")
    print(f"  - History: {obs.history}")
    print(f"  - Done: {obs.done}")
    
    # Test action validation
    if task_id == "order_status_easy":
        # Test correct action
        action = Action(action_type="lookup_order", order_id="12345")
        obs, reward, done, _ = env.step(action)
        print(f"✓ Correct action '{action.action_type}' → Reward: {reward.value}")
        
        # Test incorrect action
        action = Action(action_type="send_reply", message="Wrong action")
        obs, reward, done, _ = env.step(action)
        print(f"✓ Incorrect action → Penalty: {reward.value}")
        
    elif task_id == "refund_policy_medium":
        # Test correct action
        action = Action(action_type="send_reply", message="Our refund policy allows returns")
        obs, reward, done, _ = env.step(action)
        print(f"✓ Correct action with 'refund' → Reward: {reward.value}")
        
        # Test incorrect action
        action = Action(action_type="lookup_order", order_id="12345")
        obs, reward, done, _ = env.step(action)
        print(f"✓ Incorrect action → Penalty: {reward.value}")
        
    elif task_id == "address_change_hard":
        # Test the exact sequence
        actions = [
            Action(action_type="lookup_order", order_id="12345"),
            Action(action_type="send_reply", message="Please provide your new address"),
            Action(action_type="send_reply", message="Please confirm your new address"),
        ]
        
        total_reward = 0
        for i, action in enumerate(actions, 1):
            obs, reward, done, _ = env.step(action)
            total_reward += reward.value
            print(f"✓ Step {i}: {action.action_type} → Reward: {reward.value}")
        
        print(f"  Total reward: {total_reward}")
        
        # Verify done condition
        assert done, "Task should be done after 3 steps"
        print(f"✓ Task marked as done correctly")
        
    print(f"\n✓ Task '{task_id}' validation complete")
    return True

async def validate_file_structure():
    """Validate required files exist"""
    print(f"\n{'='*50}")
    print("Validating Project Structure")
    print(f"{'='*50}")
    
    required_files = [
        "inference.py",
        "Dockerfile",
        "openenv.yaml",
        "README.md",
        "requirements.txt",
        "app/__init__.py",
        "app/env.py",
        "app/models.py",
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️ Missing {len(missing)} required files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("\n✓ All required files present")
    return True

async def validate_openenv_spec():
    """Validate OpenEnv specification compliance"""
    print(f"\n{'='*50}")
    print("Validating OpenEnv Specification")
    print(f"{'='*50}")
    
    checks = []
    
    # Check 1: Models exist and are typed
    try:
        from app.models import Action, Observation, Reward
        checks.append(("Pydantic models exist", True))
        print("✓ Pydantic models found (Action, Observation, Reward)")
    except ImportError as e:
        checks.append(("Pydantic models exist", False))
        print(f"✗ Models missing: {e}")
    
    # Check 2: Environment implements required methods
    try:
        env = CustomerSupportEnv("order_status_easy")
        assert hasattr(env, 'reset'), "Missing reset()"
        assert hasattr(env, 'step'), "Missing step()"
        assert hasattr(env, 'state'), "Missing state()"
        checks.append(("Environment methods (reset/step/state)", True))
        print("✓ Environment implements reset(), step(), state()")
    except Exception as e:
        checks.append(("Environment methods", False))
        print(f"✗ Environment missing methods: {e}")
    
    # Check 3: Reward range validation
    try:
        env = CustomerSupportEnv("order_status_easy")
        obs = env.reset()
        action = Action(action_type="lookup_order", order_id="12345")
        obs, reward, done, _ = env.step(action)
        assert 0.0 <= reward.value <= 1.0, f"Reward {reward.value} outside [0,1]"
        checks.append(("Reward range [0,1]", True))
        print(f"✓ Rewards properly bounded: {reward.value}")
    except Exception as e:
        checks.append(("Reward range", False))
        print(f"✗ Reward range violation: {e}")
    
    # Check 4: Multiple tasks
    tasks = ["order_status_easy", "refund_policy_medium", "address_change_hard"]
    try:
        for task in tasks:
            env = CustomerSupportEnv(task)
            obs = env.reset()
            assert obs.task_id == task
        checks.append(("Multiple tasks working", True))
        print(f"✓ All {len(tasks)} tasks initialize correctly")
    except Exception as e:
        checks.append(("Multiple tasks", False))
        print(f"✗ Task initialization failed: {e}")
    
    # Check 5: Episode termination
    try:
        env = CustomerSupportEnv("order_status_easy")
        obs = env.reset()
        action = Action(action_type="lookup_order", order_id="12345")
        obs, reward, done, _ = env.step(action)
        assert done == True, "Task should be done after correct action"
        checks.append(("Episode termination", True))
        print("✓ Episodes terminate correctly")
    except Exception as e:
        checks.append(("Episode termination", False))
        print(f"✗ Episode termination issue: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("Validation Summary")
    print(f"{'='*50}")
    passed = sum(1 for _, p in checks if p)
    total = len(checks)
    print(f"Passed: {passed}/{total}")
    
    return passed == total

async def main():
    print("="*60)
    print("CUSTOMER SUPPORT ENVIRONMENT - OPENENV VALIDATION")
    print("="*60)
    
    # Validate file structure
    files_valid = await validate_file_structure()
    
    # Validate spec compliance
    spec_valid = await validate_openenv_spec()
    
    if files_valid and spec_valid:
        # Validate each task
        tasks = ["order_status_easy", "refund_policy_medium", "address_change_hard"]
        for task in tasks:
            await validate_task(task)
        
        print(f"\n{'='*60}")
        print("✅ ALL VALIDATIONS PASSED!")
        print("Environment is OpenEnv compliant and ready for submission")
        print(f"{'='*60}")
        return True
    else:
        print(f"\n{'='*60}")
        print("❌ VALIDATION FAILED")
        print("Please fix the issues above before submitting")
        print(f"{'='*60}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)