import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hybrid_sim import HybridExecutor
from simulator import RobotSimulator


class TestHybridExecutor:
    """Test hybrid executor with simulator fallback."""
    
    def test_hybrid_executor_initializes(self):
        """Executor should initialize even without hardware."""
        executor = HybridExecutor(try_hardware=False)
        assert executor is not None
        assert executor.mode == "simulator"
    
    def test_hybrid_executor_executes_move(self):
        """Should execute move command."""
        executor = HybridExecutor(try_hardware=False)
        result = executor.execute('move', 'forward', 10.0)
        
        assert result['success'] is True
        assert 'movement' in result or 'status' in result
    
    def test_hybrid_executor_executes_rotate(self):
        """Should execute rotate command."""
        executor = HybridExecutor(try_hardware=False)
        result = executor.execute('rotate', 'right', angle_deg=45.0)
        
        assert result['success'] is True
        assert 'rotation' in result or 'status' in result
    
    def test_hybrid_executor_executes_grab(self):
        """Should execute grab command."""
        executor = HybridExecutor(try_hardware=False)
        result = executor.execute('grab')
        
        assert result['success'] is True
        assert 'gripper' in result or 'status' in result
    
    def test_hybrid_executor_get_status(self):
        """Should return status with mode info."""
        executor = HybridExecutor(try_hardware=False)
        status = executor.get_status()
        
        assert 'mode' in status
        assert status['mode'] == 'simulator'
        assert 'hardware_available' in status
    
    def test_hybrid_executor_closes_cleanly(self):
        """Should close without errors."""
        executor = HybridExecutor(try_hardware=False)
        executor.close()
        
        assert executor.mode == 'simulator'
    
    def test_multiple_commands_sequence(self):
        """Should handle multiple commands in sequence."""
        executor = HybridExecutor(try_hardware=False)
        
        # Move
        result1 = executor.execute('move', 'forward', 5.0)
        assert result1['success'] is True
        
        # Rotate
        result2 = executor.execute('rotate', 'right', angle_deg=90.0)
        assert result2['success'] is True
        
        # Grab
        result3 = executor.execute('grab')
        assert result3['success'] is True
        
        # Move again
        result4 = executor.execute('move', 'backward', 5.0)
        assert result4['success'] is True
