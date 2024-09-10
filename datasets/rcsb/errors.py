
"""errors.py modified from OpenFold

----------------
[LICENSE]

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

----------------
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). 
All Bytedance's Modifications are Copyright (2024) Bytedance Ltd. and/or its affiliates. 
"""

"""Error class for handled errors."""


class DataError(Exception):
    """Data exception."""
    pass


class HeaderError(DataError):
    """Raised when failed to parse mmcif file header information."""
    pass


class NoProteinError(DataError):
    """Raised when mmcif file does not contain proteins."""
    pass


class NoValidChainError(DataError):
    """Raised when mmcif file does not contain valid protein chains."""
    pass


class ResolutionError(DataError):
    """Raised when resolution isn't acceptable."""
    pass


class ResidueError(DataError):
    """Raised when failed to locate a residue by its ID."""
    pass
