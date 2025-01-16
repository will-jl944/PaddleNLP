# coding:utf-8
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

__all__ = [
    "empty_device_cache",
]


def empty_device_cache():
    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()
    elif paddle.device.is_compiled_with_xpu():
        paddle.device.xpu.empty_cache()
    else:
        pass
