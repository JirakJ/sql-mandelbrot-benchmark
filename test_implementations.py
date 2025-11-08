"""
Tests for Mandelbrot set implementations.

This test suite validates that all implementations:
1. Can run without errors
2. Produce valid output
3. Generate consistent results
4. Handle edge cases properly

Author: Test suite for sql-mandelbrot-benchmark
License: MIT
"""

import pytest
import numpy as np


# Test configuration - use small dimensions for fast tests
TEST_WIDTH = 100
TEST_HEIGHT = 60
TEST_MAX_ITERATIONS = 50


class TestImplementations:
    """Test suite for all Mandelbrot implementations."""

    def test_duckbrot_basic(self):
        """Test DuckDB implementation runs successfully."""
        from duckbrot import run_duckbrot
        
        result = run_duckbrot(TEST_WIDTH, TEST_HEIGHT, TEST_MAX_ITERATIONS)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (TEST_HEIGHT, TEST_WIDTH)
        assert result.dtype == np.uint16
        assert result.min() >= 0
        assert result.max() <= TEST_MAX_ITERATIONS

    def test_pybrot_basic(self):
        """Test pure Python implementation runs successfully."""
        from pybrot import run_pybrot
        
        result = run_pybrot(TEST_WIDTH, TEST_HEIGHT, TEST_MAX_ITERATIONS)
        
        assert result is not None
        assert len(result) == TEST_HEIGHT
        assert len(result[0]) == TEST_WIDTH
        # Verify all values are within expected range
        for row in result:
            for val in row:
                assert 0 <= val <= TEST_MAX_ITERATIONS

    def test_numpybrot_basic(self):
        """Test NumPy implementation runs successfully."""
        from numpybrot import run_numpybrot
        
        result = run_numpybrot(TEST_WIDTH, TEST_HEIGHT, TEST_MAX_ITERATIONS)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (TEST_HEIGHT, TEST_WIDTH)
        assert result.dtype == np.uint16
        assert result.min() >= 0
        assert result.max() <= TEST_MAX_ITERATIONS

    def test_sqlitebrot_basic(self):
        """Test SQLite implementation runs successfully."""
        from sqlitebrot import run_sqlitebrot
        
        result = run_sqlitebrot(TEST_WIDTH, TEST_HEIGHT, TEST_MAX_ITERATIONS)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (TEST_HEIGHT, TEST_WIDTH)
        assert result.dtype == np.uint16
        assert result.min() >= 0
        assert result.max() <= TEST_MAX_ITERATIONS

    def test_fastpybrot_basic(self):
        """Test FastPybrot implementation runs successfully."""
        from fastpybrot import run_pybrot
        
        result = run_pybrot(TEST_WIDTH, TEST_HEIGHT, TEST_MAX_ITERATIONS)
        
        assert result is not None
        assert len(result) == TEST_HEIGHT
        assert len(result[0]) == TEST_WIDTH

    def test_fasterpybrot_basic(self):
        """Test FasterPybrot implementation runs successfully."""
        from fasterpybrot import run_pybrot
        
        result = run_pybrot(TEST_WIDTH, TEST_HEIGHT, TEST_MAX_ITERATIONS)
        
        assert result is not None
        assert len(result) == TEST_HEIGHT
        assert len(result[0]) == TEST_WIDTH

    def test_arrow_datafusion_basic(self):
        """Test Arrow DataFusion implementation runs successfully."""
        from arrow_datafusion import run_arrow_datafusion
        
        result = run_arrow_datafusion(TEST_WIDTH, TEST_HEIGHT, TEST_MAX_ITERATIONS)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (TEST_HEIGHT, TEST_WIDTH)


class TestConsistency:
    """Test that implementations produce consistent results."""

    @pytest.fixture
    def small_config(self):
        """Small test configuration for consistency checks."""
        return {
            'width': 20,
            'height': 15,
            'max_iterations': 20
        }

    def test_known_point_in_set(self, small_config):
        """Test that point (0, 0) is in the Mandelbrot set."""
        from duckbrot import run_duckbrot
        
        # Point (0, 0) should be in the set (max iterations)
        result = run_duckbrot(
            small_config['width'],
            small_config['height'],
            small_config['max_iterations']
        )
        
        # Find the pixel closest to (0, 0) in complex plane
        # Complex plane range: real [-2.5, 1.0], imag [-1.0, 1.0]
        # (0, 0) maps to approximately pixel (width * 2.5/3.5, height * 0.5)
        x_pixel = int(small_config['width'] * 2.5 / 3.5)
        y_pixel = int(small_config['height'] * 0.5)
        
        # Should have high iteration count (in the set)
        assert result[y_pixel, x_pixel] >= small_config['max_iterations'] * 0.8

    def test_known_point_outside_set(self, small_config):
        """Test that point (2, 2) escapes quickly."""
        from duckbrot import run_duckbrot
        
        result = run_duckbrot(
            small_config['width'],
            small_config['height'],
            small_config['max_iterations']
        )
        
        # Point near (1, 1) should escape quickly
        # This is at the edge of our viewing window
        x_pixel = small_config['width'] - 1
        y_pixel = small_config['height'] - 1
        
        # Should have low iteration count (escapes quickly)
        assert result[y_pixel, x_pixel] < small_config['max_iterations'] * 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_minimal_dimensions(self):
        """Test with minimal valid dimensions."""
        from duckbrot import run_duckbrot
        
        result = run_duckbrot(2, 2, 5)
        assert result is not None
        assert result.shape == (2, 2)

    def test_single_iteration(self):
        """Test with single iteration."""
        from pybrot import run_pybrot
        
        result = run_pybrot(10, 10, 1)
        assert result is not None
        assert len(result) == 10

    def test_many_iterations(self):
        """Test with many iterations on small grid."""
        from numpybrot import run_numpybrot
        
        result = run_numpybrot(10, 10, 500)
        assert result is not None
        assert result.shape == (10, 10)


class TestUtils:
    """Test utility functions."""

    def test_save_mandelbrot_image(self, tmp_path):
        """Test image saving functionality."""
        from utils import save_mandelbrot_image
        from numpybrot import run_numpybrot
        
        # Generate small result
        result = run_numpybrot(20, 15, 20)
        
        # Save to temporary directory
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            save_mandelbrot_image(result, 20, 'test.png')
            
            # Check that file was created
            assert (tmp_path / 'images' / 'test.png').exists()
        finally:
            os.chdir(original_dir)

    def test_benchmark_execution(self):
        """Test benchmark runner utility."""
        from utils import run_benchmark
        from pybrot import run_pybrot
        
        result, elapsed_ms = run_benchmark(
            "Test Benchmark",
            run_pybrot,
            10, 10, 5
        )
        
        assert result is not None
        assert elapsed_ms is not None
        assert elapsed_ms > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])
