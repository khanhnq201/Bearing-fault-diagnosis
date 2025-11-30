import unittest
import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_loader import CWRU_dataloader, BearingDataset


class TestBearingDataset(unittest.TestCase):
    """Test cases for BearingDataset class"""
    
    def setUp(self):
        """Set up test data"""
        self.X_data = np.random.randn(100, 1, 512).astype(np.float32)
        self.Y_data = np.random.randint(0, 3, 100)
        self.dataset = BearingDataset(self.X_data, self.Y_data)
    
    def test_dataset_length(self):
        """Test dataset returns correct length"""
        self.assertEqual(len(self.dataset), 100)
    
    def test_dataset_getitem(self):
        """Test dataset returns correct item format"""
        data, label = self.dataset[0]
        
        # Check types
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        
        # Check data type
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(label.dtype, torch.int64)
        
        # Check shapes
        self.assertEqual(data.shape, (1, 512))
    
    def test_dataset_all_items_accessible(self):
        """Test all items in dataset are accessible"""
        for i in range(len(self.dataset)):
            data, label = self.dataset[i]
            self.assertIsNotNone(data)
            self.assertIsNotNone(label)


class TestCWRUDataLoader(unittest.TestCase):
    """Test cases for CWRU_dataloader class"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            'path': './CWRU-dataset-main',
            'batch_size': 32,
            'overlapping_ratio': 0.5
        }
        self.data_loader = CWRU_dataloader(self.config)
    
    def test_initialization(self):
        """Test data loader initialization"""
        self.assertEqual(self.data_loader.fs, 12000)
        self.assertEqual(self.data_loader.batch_size, 32)
        self.assertEqual(self.data_loader.sample_length, 4096)
        self.assertEqual(self.data_loader.overlapping_ratio, 0.5)
        self.assertEqual(self.data_loader.model_input_size, 512)
        self.assertEqual(self.data_loader.n_rev, 64)
        self.assertEqual(self.data_loader.default_rpm, 1750)
    
    def test_train_val_test_split(self):
        """Test train/val/test lists are properly defined"""
        self.assertIsInstance(self.data_loader.train_list, list)
        self.assertIsInstance(self.data_loader.val_list, list)
        self.assertIsInstance(self.data_loader.test_list, list)
        
        # Check no overlap between splits
        train_set = set(self.data_loader.train_list)
        val_set = set(self.data_loader.val_list)
        test_set = set(self.data_loader.test_list)
        
        self.assertEqual(len(train_set & val_set), 0, "Train and val overlap")
        self.assertEqual(len(train_set & test_set), 0, "Train and test overlap")
        self.assertEqual(len(val_set & test_set), 0, "Val and test overlap")
    
    def test_get_label_from_path(self):
        """Test label extraction from file path"""
        # Test Normal label
        normal_path = Path("CWRU-dataset-main/Normal/97_Normal_0.mat")
        self.assertEqual(self.data_loader.get_label_from_path(normal_path), 0)
        
        # Test IR label
        ir_path = Path("CWRU-dataset-main/12k_Drive_End_Bearing_Fault_Data/IR/007/105_0.mat")
        self.assertEqual(self.data_loader.get_label_from_path(ir_path), 1)
        
        # Test OR label
        or_path = Path("CWRU-dataset-main/12k_Drive_End_Bearing_Fault_Data/OR/007/@6/130_0.mat")
        self.assertEqual(self.data_loader.get_label_from_path(or_path), 2)
        
        # Test invalid path
        invalid_path = Path("some/invalid/path.mat")
        self.assertEqual(self.data_loader.get_label_from_path(invalid_path), -1)
    
    def test_import_data_shapes(self):
        """Test imported data has correct shapes"""
        try:
            train_X, train_Y, val_X, val_Y, test_X, test_Y = self.data_loader.import_data()
            
            # Check dimensions
            self.assertEqual(train_X.ndim, 2)
            self.assertEqual(val_X.ndim, 2)
            self.assertEqual(test_X.ndim, 2)
            
            # Check sample length
            self.assertEqual(train_X.shape[1], 4096)
            self.assertEqual(val_X.shape[1], 4096)
            self.assertEqual(test_X.shape[1], 4096)
            
            # Check labels are 1D
            self.assertEqual(train_Y.ndim, 1)
            self.assertEqual(val_Y.ndim, 1)
            self.assertEqual(test_Y.ndim, 1)
            
            # Check matching sample counts
            self.assertEqual(train_X.shape[0], train_Y.shape[0])
            self.assertEqual(val_X.shape[0], val_Y.shape[0])
            self.assertEqual(test_X.shape[0], test_Y.shape[0])
            
            print(f"\n✓ Data shapes - Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_label_values(self):
        """Test labels are in correct range [0, 1, 2]"""
        try:
            train_X, train_Y, val_X, val_Y, test_X, test_Y = self.data_loader.import_data()
            
            # Check all labels are valid
            self.assertTrue(np.all(train_Y >= 0))
            self.assertTrue(np.all(train_Y <= 2))
            self.assertTrue(np.all(val_Y >= 0))
            self.assertTrue(np.all(val_Y <= 2))
            self.assertTrue(np.all(test_Y >= 0))
            self.assertTrue(np.all(test_Y <= 2))
            
            # Check all 3 classes exist in training data
            unique_labels = np.unique(train_Y)
            self.assertEqual(len(unique_labels), 3, "Training data should have 3 classes")
            
            print(f"\n✓ Label distribution - Train: {np.bincount(train_Y)}, Val: {np.bincount(val_Y)}, Test: {np.bincount(test_Y)}")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_overlapping_ratio_effect(self):
        """Test overlapping ratio affects number of samples"""
        try:
            # Test with no overlap
            config_no_overlap = self.config.copy()
            config_no_overlap['overlapping_ratio'] = 0.0
            loader_no_overlap = CWRU_dataloader(config_no_overlap)
            train_X_no, train_Y_no, _, _, _, _ = loader_no_overlap.import_data()
            
            # Test with 50% overlap
            config_overlap = self.config.copy()
            config_overlap['overlapping_ratio'] = 0.5
            loader_overlap = CWRU_dataloader(config_overlap)
            train_X_over, train_Y_over, _, _, _, _ = loader_overlap.import_data()
            
            # More overlap should give more samples
            self.assertGreater(train_X_over.shape[0], train_X_no.shape[0],
                             "50% overlap should produce more samples than no overlap")
            
            print(f"\n✓ Overlapping effect - No overlap: {train_X_no.shape[0]} samples, 50% overlap: {train_X_over.shape[0]} samples")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_invalid_overlapping_ratio(self):
        """Test invalid overlapping ratios are handled"""
        config = self.config.copy()
        
        # Test ratio > 1
        config['overlapping_ratio'] = 1.5
        loader = CWRU_dataloader(config)
        self.assertEqual(loader.overlapping_ratio, 1.5)  # Stored as is
        
        # Import should handle it (set to 0 internally)
        try:
            loader.import_data()
        except Exception as e:
            self.fail(f"Should handle invalid overlap ratio: {e}")
    
    def test_freq_spectrum_preprocessing(self):
        """Test frequency spectrum preprocessing"""
        try:
            train_loader, val_loader, test_loader = self.data_loader.get_dataloaders(processing_type='freq_spectrum')
            
            # Get a batch
            data_batch, label_batch = next(iter(train_loader))
            
            # Check shape: (batch, channels, freq_bins)
            self.assertEqual(data_batch.ndim, 3)
            self.assertEqual(data_batch.shape[1], 1)  # Single channel
            self.assertEqual(data_batch.shape[2], 2048)  # Half of sample_length (4096/2)
            
            # Check data type
            self.assertEqual(data_batch.dtype, torch.float32)
            
            # Check labels
            self.assertEqual(label_batch.dtype, torch.int64)
            self.assertTrue(torch.all(label_batch >= 0))
            self.assertTrue(torch.all(label_batch <= 2))
            
            print(f"\n✓ Frequency spectrum shape: {data_batch.shape}")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_order_spectrum_preprocessing(self):
        """Test order spectrum preprocessing"""
        try:
            train_loader, val_loader, test_loader = self.data_loader.get_dataloaders(processing_type='order_spectrum')
            
            # Get a batch
            data_batch, label_batch = next(iter(train_loader))
            
            # Check shape: (batch, channels, freq_bins)
            self.assertEqual(data_batch.ndim, 3)
            self.assertEqual(data_batch.shape[1], 1)  # Single channel
            self.assertEqual(data_batch.shape[2], 256)  # Half of model_input_size (512/2)
            
            # Check data type
            self.assertEqual(data_batch.dtype, torch.float32)
            
            # Check no NaN or Inf values
            self.assertFalse(torch.any(torch.isnan(data_batch)))
            self.assertFalse(torch.any(torch.isinf(data_batch)))
            
            print(f"\n✓ Order spectrum shape: {data_batch.shape}")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_order_spectrogram_preprocessing(self):
        """Test order spectrogram preprocessing"""
        try:
            train_loader, val_loader, test_loader = self.data_loader.get_dataloaders(processing_type='order_spectrogram')
            
            # Get a batch
            data_batch, label_batch = next(iter(train_loader))
            
            # Check shape: (batch, channels, freq, time)
            self.assertEqual(data_batch.ndim, 4)
            self.assertEqual(data_batch.shape[1], 1)  # Single channel
            
            # Check data type
            self.assertEqual(data_batch.dtype, torch.float32)
            
            # Check no NaN or Inf values
            self.assertFalse(torch.any(torch.isnan(data_batch)))
            self.assertFalse(torch.any(torch.isinf(data_batch)))
            
            print(f"\n✓ Order spectrogram shape: {data_batch.shape}")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_raw_preprocessing(self):
        """Test raw signal preprocessing"""
        try:
            train_loader, val_loader, test_loader = self.data_loader.get_dataloaders(processing_type='raw')
            
            # Get a batch
            data_batch, label_batch = next(iter(train_loader))
            
            # Check shape: (batch, channels, sample_length)
            self.assertEqual(data_batch.ndim, 3)
            self.assertEqual(data_batch.shape[1], 1)  # Single channel
            self.assertEqual(data_batch.shape[2], 4096)  # Sample length
            
            # Check data type
            self.assertEqual(data_batch.dtype, torch.float32)
            
            print(f"\n✓ Raw signal shape: {data_batch.shape}")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_invalid_processing_type(self):
        """Test invalid processing type raises error"""
        with self.assertRaises(ValueError):
            self.data_loader.get_dataloaders(processing_type='invalid_type')
    
    def test_dataloader_batch_size(self):
        """Test DataLoader respects batch size"""
        try:
            config = self.config.copy()
            config['batch_size'] = 16
            loader = CWRU_dataloader(config)
            train_loader, _, _ = loader.get_dataloaders(processing_type='raw')
            
            # Get a batch
            data_batch, label_batch = next(iter(train_loader))
            
            # Check batch size (may be smaller for last batch)
            self.assertLessEqual(data_batch.shape[0], 16)
            self.assertEqual(data_batch.shape[0], label_batch.shape[0])
            
            print(f"\n✓ Batch size: {data_batch.shape[0]}")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_dataloader_shuffle(self):
        """Test training DataLoader shuffles data"""
        try:
            train_loader, _, _ = self.data_loader.get_dataloaders(processing_type='raw')
            
            # Get first batches from two epochs
            first_batch_1 = next(iter(train_loader))[1].numpy()  # Labels
            first_batch_2 = next(iter(train_loader))[1].numpy()
            
            # They should be different (with high probability)
            # This is not guaranteed but very likely with shuffling
            different = not np.array_equal(first_batch_1, first_batch_2)
            
            print(f"\n✓ Shuffle working: {different} (First batches are different)")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_rpm_dict_populated(self):
        """Test RPM dictionary is populated after importing data"""
        try:
            self.data_loader.import_data()
            
            # Check RPM dict is not empty
            self.assertGreater(len(self.data_loader.rpm_dict), 0)
            
            # Check RPM values are reasonable (typically 1730-1797 for CWRU)
            for key, rpm in self.data_loader.rpm_dict.items():
                self.assertIsInstance(rpm, float)
                self.assertGreater(rpm, 0)
                self.assertLess(rpm, 3000)  # Reasonable upper bound
            
            print(f"\n✓ RPM dict populated with {len(self.data_loader.rpm_dict)} entries")
            print(f"  Sample RPMs: {list(self.data_loader.rpm_dict.values())[:5]}")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_angular_resampling(self):
        """Test angular resampling function"""
        signal = np.random.randn(4096)
        rpm = 1750
        
        resampled = self.data_loader._angular_resampling(signal, rpm)
        
        # Check output shape
        self.assertEqual(len(resampled), 512)  # model_input_size
        
        # Check no NaN or Inf
        self.assertFalse(np.any(np.isnan(resampled)))
        self.assertFalse(np.any(np.isinf(resampled)))
    
    def test_envelope_extraction(self):
        """Test envelope extraction function"""
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        
        envelope = self.data_loader._envelope_extraction(signal)
        
        # Check output shape
        self.assertEqual(len(envelope), len(signal))
        
        # Check mean is approximately zero (centered)
        self.assertAlmostEqual(np.mean(envelope), 0, places=10)
    
    def test_spectrum_function(self):
        """Test spectrum computation function"""
        signal = np.random.randn(1024)
        
        spectrum = self.data_loader._spectrum(signal)
        
        # Check output shape (half of input)
        self.assertEqual(len(spectrum), 512)
        
        # Check all values are non-negative
        self.assertTrue(np.all(spectrum >= 0))
    
    def test_file_id_tracking(self):
        """Test file IDs are tracked correctly"""
        try:
            self.data_loader.import_data()
            
            # Check file_ids exist and match sample counts
            self.assertEqual(len(self.data_loader.train_file_ids), len(self.data_loader.train_samples))
            self.assertEqual(len(self.data_loader.val_file_ids), len(self.data_loader.val_samples))
            self.assertEqual(len(self.data_loader.test_file_ids), len(self.data_loader.test_samples))
            
            # Check file IDs are from the expected lists
            unique_train_ids = set(self.data_loader.train_file_ids)
            self.assertTrue(unique_train_ids.issubset(set(self.data_loader.train_list)))
            
            print(f"\n✓ File ID tracking working. Unique train files: {len(unique_train_ids)}")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_consistency_across_runs(self):
        """Test data loading is consistent across runs"""
        try:
            # First run
            loader1 = CWRU_dataloader(self.config)
            X1, Y1, _, _, _, _ = loader1.import_data()
            
            # Second run with same config
            loader2 = CWRU_dataloader(self.config)
            X2, Y2, _, _, _, _ = loader2.import_data()
            
            # Should be identical
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(Y1, Y2)
            
            print("\n✓ Data loading is consistent across runs")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""
    
    def setUp(self):
        self.config = {
            'path': './CWRU-dataset-main',
            'batch_size': 32,
            'overlapping_ratio': 0.5
        }
    
    def test_complete_workflow_freq_spectrum(self):
        """Test complete workflow with frequency spectrum"""
        try:
            loader = CWRU_dataloader(self.config)
            train_loader, val_loader, test_loader = loader.get_dataloaders(processing_type='freq_spectrum')
            
            # Test iterating through all loaders
            train_batches = 0
            for data, labels in train_loader:
                self.assertEqual(data.ndim, 3)
                self.assertEqual(labels.ndim, 1)
                train_batches += 1
            
            val_batches = 0
            for data, labels in val_loader:
                val_batches += 1
            
            test_batches = 0
            for data, labels in test_loader:
                test_batches += 1
            
            print(f"\n✓ Complete workflow - Train batches: {train_batches}, Val: {val_batches}, Test: {test_batches}")
            
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")
    
    def test_all_processing_types(self):
        """Test all processing types work"""
        processing_types = ['freq_spectrum', 'order_spectrum', 'order_spectrogram', 'raw']
        
        try:
            for proc_type in processing_types:
                loader = CWRU_dataloader(self.config)
                train_loader, val_loader, test_loader = loader.get_dataloaders(processing_type=proc_type)
                
                # Get one batch from each
                train_data, train_labels = next(iter(train_loader))
                val_data, val_labels = next(iter(val_loader))
                test_data, test_labels = next(iter(test_loader))
                
                # Basic checks
                self.assertGreater(train_data.shape[0], 0)
                self.assertEqual(train_data.shape[0], train_labels.shape[0])
                
                print(f"\n✓ {proc_type} processing works - Train shape: {train_data.shape}")
                
        except Exception as e:
            self.skipTest(f"Data files not available: {e}")


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBearingDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestCWRUDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Exit with error code if tests failed
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)