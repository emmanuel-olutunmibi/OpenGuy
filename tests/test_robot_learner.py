"""
test_robot_learner.py - Tests for robot learning system
"""

import pytest
import json
import tempfile
from pathlib import Path
from robot_learner import RobotLearner, CommandExperience, AdaptiveStrategy


class TestCommandExperience:
    """Test CommandExperience class."""
    
    def test_create_success_experience(self):
        """Test creating a successful experience."""
        exp = CommandExperience("move", "forward", 50, None, True, execution_time=1.5)
        
        assert exp.action == "move"
        assert exp.direction == "forward"
        assert exp.distance == 50
        assert exp.success is True
        assert exp.error is None
        assert exp.execution_time == 1.5
    
    def test_create_failure_experience(self):
        """Test creating a failed experience."""
        exp = CommandExperience("move", "forward", 50, None, False, 
                               error="COLLISION_DETECTED", notes="Hit wall")
        
        assert exp.success is False
        assert exp.error == "COLLISION_DETECTED"
        assert exp.notes == "Hit wall"
    
    def test_to_dict_conversion(self):
        """Test converting experience to dictionary."""
        exp = CommandExperience("rotate", "left", None, 45, True)
        data = exp.to_dict()
        
        assert data["action"] == "rotate"
        assert data["direction"] == "left"
        assert data["angle"] == 45
        assert data["success"] is True
        assert "timestamp" in data
    
    def test_from_dict_conversion(self):
        """Test creating experience from dictionary."""
        original = CommandExperience("move", "forward", 25, None, True, 
                                    execution_time=0.8, notes="Test")
        data = original.to_dict()
        
        restored = CommandExperience.from_dict(data)
        
        assert restored.action == original.action
        assert restored.distance == original.distance
        assert restored.success == original.success


class TestAdaptiveStrategy:
    """Test AdaptiveStrategy class."""
    
    def test_create_strategy(self):
        """Test creating a strategy."""
        strategy = AdaptiveStrategy("move", "forward", 50, None)
        
        assert strategy.action == "move"
        assert strategy.direction == "forward"
        assert strategy.success_rate == 0.5  # Unknown
        assert strategy.total_attempts == 0
    
    def test_track_success_rate(self):
        """Test tracking success rate."""
        strategy = AdaptiveStrategy("move", "forward", 50, None)
        
        # Record 10 attempts: 8 successes, 2 failures
        for i in range(8):
            strategy.record_attempt(success=True, time_taken=1.0)
        for i in range(2):
            strategy.record_attempt(success=False, time_taken=0)
        
        assert strategy.total_attempts == 10
        assert strategy.successful_attempts == 8
        assert strategy.success_rate == 0.8
    
    def test_track_failure_reasons(self):
        """Test tracking failure reasons."""
        strategy = AdaptiveStrategy("move", "forward", 50, None)
        
        strategy.record_attempt(success=False, time_taken=0, error="COLLISION")
        strategy.record_attempt(success=False, time_taken=0, error="COLLISION")
        strategy.record_attempt(success=False, time_taken=0, error="TIMEOUT")
        
        assert strategy.failure_reasons["COLLISION"] == 2
        assert strategy.failure_reasons["TIMEOUT"] == 1
    
    def test_should_reduce_step_size_when_failing(self):
        """Test that strategy recommends smaller steps when failing."""
        strategy = AdaptiveStrategy("move", "forward", 50, None)
        
        # Simulate poor success rate
        for i in range(10):
            strategy.record_attempt(success=i % 3 == 0, time_taken=1.0)  # ~33% success
        
        assert strategy.should_reduce_step_size() is True
    
    def test_update_strategy_reduces_parameters_on_failure(self):
        """Test strategy adaptation reduces parameters when failing."""
        strategy = AdaptiveStrategy("move", "forward", 50, None)
        
        # Simulate many failures
        for i in range(15):
            strategy.record_attempt(success=i > 12, time_taken=1.0)  # ~13% success
        
        strategy.update_strategy()
        
        assert strategy.recommended_distance_adjustment < 1.0
        assert strategy.recommended_speed < 1.0
    
    def test_update_strategy_increases_parameters_on_success(self):
        """Test strategy adaptation increases parameters when succeeding."""
        strategy = AdaptiveStrategy("move", "forward", 50, None)
        
        # Simulate great success (20 successes, 0 failures - 100%)
        for i in range(20):
            strategy.record_attempt(success=True, time_taken=1.0)
        
        strategy.update_strategy()
        
        assert strategy.recommended_distance_adjustment > 1.0
        assert strategy.recommended_speed > 1.0


class TestRobotLearner:
    """Test RobotLearner main class."""
    
    @pytest.fixture
    def learner(self, tmp_path):
        """Create a learner with temporary directory."""
        return RobotLearner("test_robot", str(tmp_path / "learning"))
    
    def test_initialize_learner(self, learner):
        """Test initializing a learner."""
        assert learner.robot_id == "test_robot"
        assert learner.learn_dir.exists()
        assert len(learner.experiences) == 0
        assert len(learner.strategies) == 0
    
    def test_record_experience_success(self, learner):
        """Test recording a successful experience."""
        learner.record_experience("move", "forward", 50, None, True, execution_time=1.0)
        
        assert len(learner.experiences) == 1
        assert learner.experiences[0].success is True
        assert len(learner.strategies) == 1
    
    def test_record_experience_failure(self, learner):
        """Test recording a failed experience."""
        learner.record_experience("move", "forward", 50, None, False, 
                                 error="COLLISION_DETECTED")
        
        assert len(learner.experiences) == 1
        assert learner.experiences[0].success is False
        assert learner.experiences[0].error == "COLLISION_DETECTED"
    
    def test_multiple_experiences_update_strategy(self, learner):
        """Test that multiple experiences update the strategy."""
        # Record 10 attempts
        for i in range(10):
            success = i < 8  # 80% success rate
            learner.record_experience("move", "forward", 50, None, success)
        
        strategy = learner.strategies["move_forward"]
        assert strategy.total_attempts == 10
        assert strategy.success_rate == 0.8
    
    def test_get_adaptive_parameters_unknown_command(self, learner):
        """Test getting parameters for unknown command."""
        params = learner.get_adaptive_parameters("move", "forward", 50, None)
        
        assert params["distance"] == 50
        assert params["step_size_multiplier"] == 1.0
        assert params["confidence"] == 0.5  # Unknown
    
    def test_get_adaptive_parameters_learned_command(self, learner):
        """Test getting parameters for learned command."""
        # Train: 8 successes out of 10
        for i in range(10):
            learner.record_experience("move", "forward", 50, None, i < 8)
        
        params = learner.get_adaptive_parameters("move", "forward", 50, None)
        
        assert "distance" in params
        assert "confidence" in params
        assert params["confidence"] == 0.8
    
    def test_register_recovery_strategy(self, learner):
        """Test registering a recovery strategy."""
        learner.register_recovery_strategy("move", "forward", "FALL_DETECTED",
                                          "reduce_speed_fifty_percent")
        
        recovery = learner.get_recovery_strategy("move", "forward", "FALL_DETECTED")
        assert recovery == "reduce_speed_fifty_percent"
    
    def test_get_best_command_variants(self, learner):
        """Test getting best command variants."""
        # Record same command with different distances
        learner.record_experience("move", "forward", 30, None, True)
        learner.record_experience("move", "forward", 30, None, True)
        learner.record_experience("move", "forward", 50, None, True)
        learner.record_experience("move", "forward", 50, None, False)
        
        variants = learner.get_best_command_variants("move", "forward", 2)
        
        assert len(variants) <= 2
        # 30cm should appear more often in successful attempts
        if len(variants) > 0:
            assert variants[0]["distance"] == 30
            assert variants[0]["success_count"] == 2
    
    def test_get_learning_report(self, learner):
        """Test getting a learning report."""
        # Record some experiences
        learner.record_experience("move", "forward", 50, None, True)
        learner.record_experience("move", "forward", 50, None, True)
        learner.record_experience("move", "backward", 50, None, False)
        learner.record_experience("rotate", "left", None, 45, True)
        
        report = learner.get_learning_report()
        
        assert report["robot_id"] == "test_robot"
        assert report["total_experiences"] == 4
        assert report["total_successes"] == 3
        assert "learned_strategies" in report
        assert len(report["strategies"]) == 3
    
    def test_save_and_load_models(self, learner):
        """Test saving and loading learned models."""
        # Record experiences
        learner.record_experience("move", "forward", 50, None, True)
        learner.record_experience("move", "forward", 50, None, False, error="COLLISION")
        learner.record_experience("rotate", "left", None, 45, True)
        
        # Save
        learner._save_models()
        
        # Load into new learner
        learner2 = RobotLearner("test_robot", str(learner.learn_dir.parent / "learning"))
        
        # Verify loaded data
        assert len(learner2.strategies) == 2
        assert "move_forward" in learner2.strategies
        assert "rotate_left" in learner2.strategies
    
    def test_export_learned_code(self, learner):
        """Test exporting learned behavior as code."""
        learner.record_experience("move", "forward", 50, None, True)
        learner.record_experience("rotate", "left", None, 45, True)
        
        code = learner.export_learned_code()
        
        assert "LEARNED_PARAMETERS" in code
        assert "move_forward" in code
        assert "rotate_left" in code
        assert "success_rate" in code
        assert "def get_parameters" in code


class TestRobotLearnerIntegration:
    """Integration tests for robot learning."""
    
    @pytest.fixture
    def learner(self, tmp_path):
        """Create a learner with temporary directory."""
        return RobotLearner("integration_test", str(tmp_path / "learning"))
    
    def test_learn_from_repeated_success(self, learner):
        """Test that robot learns from repeated success."""
        # First time: uncertain
        params_initial = learner.get_adaptive_parameters("move", "forward", 50, None)
        assert params_initial["confidence"] == 0.5
        
        # Record many successes
        for i in range(20):
            learner.record_experience("move", "forward", 50, None, True, execution_time=1.0)
        
        # Now we're confident
        params_final = learner.get_adaptive_parameters("move", "forward", 50, None)
        assert params_final["confidence"] == 1.0
        assert params_final["step_size_multiplier"] > 1.0  # Can be more aggressive
    
    def test_recover_from_repeated_failure(self, learner):
        """Test that robot learns to be cautious after failures."""
        # Record many failures
        for i in range(15):
            learner.record_experience("move", "forward", 100, None, False, 
                                     error="COLLISION")
        
        params = learner.get_adaptive_parameters("move", "forward", 100, None)
        
        # Should recommend smaller steps
        assert params["step_size_multiplier"] < 1.0
        assert params["break_into_steps"] is True
        assert params["recommended_steps"] > 1
    
    def test_mixed_success_and_failure(self, learner):
        """Test learning with mixed results."""
        # 70% success rate
        for i in range(100):
            learner.record_experience("move", "forward", 50, None, i % 10 < 7)
        
        report = learner.get_learning_report()
        strategy = learner.strategies["move_forward"]
        
        assert 0.65 < strategy.success_rate < 0.75
        assert strategy.total_attempts == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
