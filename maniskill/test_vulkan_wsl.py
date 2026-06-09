"""Probe SAPIEN Vulkan ICD setup in WSL."""
import os
print('VK_ICD_FILENAMES before sapien import:', os.environ.get('VK_ICD_FILENAMES', '(unset)'))

# Force sapien to attempt ICD setup
import sapien
print('VK_ICD_FILENAMES after sapien import:', os.environ.get('VK_ICD_FILENAMES', '(unset)'))

import sapien.render as sr
print('GPU summary:')
print(sr.get_device_summary())
