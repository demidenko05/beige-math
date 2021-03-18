#!/usr/bin/env python
# coding=UTF-8

# Copyright (c) 2000-2019 Chih-Chung Chang and Chih-Jen Lin
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither name of copyright holders nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.


# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#Code from LIBSVM original library

import cmath

#from cvm.cpp# double Kernel::k_function(const svm_node *x, const svm_node *y, const svm_parameter& param)
#gamma by default 1/vector dimension
#here vectors are the same size
class LibSvmOrigKernRbf:
  def __init__(self, pGamma):
    self.gamma = pGamma
  
  #passed data must be preliminary validated! 
  #return dot product of two vectors
  def dot (self, pVEC1, pVEC2):
    SP = pVEC1 - pVEC2
    sz = SP.shape[0]
    sm = 0.0
    for i in range (sz):
      sm += SP[i] * SP[i]
    for i in range (sz):
      sm += pVEC1[i] * pVEC1[i]
    for i in range (sz):
      sm += pVEC2[i] * pVEC2[i]
    return cmath.e ** (- self.gamma * sm)
