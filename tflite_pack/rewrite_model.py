#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of TFLitePack.
# See https://github.com/tum-ei-eda/tflite-pack.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
#

import os
import logging
import argparse

# import struct

import flatbuffers

import tflite as tflite_original
import tumeda_tflite as tflite
import tumeda_tflite.Model
import tumeda_tflite.QuantizationDetails
import tumeda_tflite.CustomQuantization

# from tumeda_tflite.BuiltinOperator import BuiltinOperator as OpType
# from tumeda_tflite.TensorType import TensorType as TType
from tumeda_tflite.Operator import OperatorT
import numpy as np


logging.basicConfig(format="[%(asctime)s]::%(pathname)s:%(lineno)d::%(levelname)s - %(message)s", level=logging.DEBUG)


class TfLiteRewrite:
    """TfLite Flatbuffer Rewriter"""

    def loadModelFromBuf(buf):
        """Load tflite flatbuffer to object tree"""
        model = tflite.Model.Model.GetRootAsModel(buf, 0)
        return tflite.Model.ModelT.InitFromObj(model)

    def loadModelFromFile(filename):
        """Load tflite flatbuffer file to object tree"""
        with open(filename, "rb") as f:
            buf = bytearray(f.read())
        return TfLiteRewrite.loadModelFromBuf(buf)

    def saveModelToBuf(self):
        """Save object tree to tflite flatbuffer"""
        b = flatbuffers.Builder(1024)
        b.Finish(self.modelT.Pack(b), file_identifier=b"TFL3")
        return b.Output()

    def saveModelToFile(self, filename):
        """Save object tree to tflite flatbuffer file"""
        with open(filename, "wb") as f:
            f.write(self.saveModelToBuf())

    def __init__(self, model):
        """Constructor of TfLiteRewrite"""
        logging.debug("Initializing TfLite Flatbuffer Rewriter...")
        if isinstance(model, tflite_original.Model):
            casted_model = tflite.Model.Model()
            casted_model._tab = model._tab
            self.modelT = tflite.Model.ModelT.InitFromObj(casted_model)
        elif type(model) is str:
            self.modelT = TfLiteRewrite.loadModelFromFile(model)
        elif type(model) is bytearray:
            self.modelT = TfLiteRewrite.loadModelFromBuf(model)
        elif type(model) is bytes:
            self.modelT = TfLiteRewrite.loadModelFromBuf(bytearray(model))
        else:
            raise RuntimeError("Model has to be of type str (filename) or bytearray")

    # def applyPacking(self):
    #     """ Packs all eligible tensors in the given model (originally written by Rafael Stahl) """

    #     def packTensor(t):
    #         """ Apply packing to tensor (originally written by Rafael Stahl) """
    #         PACKED_SUB8BIT_UNIFORM_DETAILS_MAGIC = 0xa4592d92
    #         # TODO get number of pack bits automatically or allow configuration
    #         bitsPerItem = 4
    #         containerBits = 8
    #         packedMinorDims = 1
    #         signedData = True

    #         if signedData:
    #             minVal = -(bitsPerItem - 1)**2
    #             maxVal = (bitsPerItem - 1)**2 - 1
    #         else:
    #             minVal = 0
    #             maxVal = bitsPerItem**2 - 1

    #         # Serialize custom quantization data.
    #         data = struct.pack("I", PACKED_SUB8BIT_UNIFORM_DETAILS_MAGIC)
    #         data += struct.pack("B", bitsPerItem)
    #         data += struct.pack("B", containerBits)
    #         data += struct.pack("B", packedMinorDims)
    #         data += b"\x00\x00\x00\x00\x00"

    #         # Edit quantization metadata to represent packed data.
    #         customQuant = tflite.CustomQuantization.CustomQuantizationT()
    #         customQuant.custom = [c for c in data]
    #         t.quantization.detailsType = QDetails.CustomQuantization
    #         t.quantization.details = customQuant

    #         # Determine run length.
    #         packingRunLength = 1
    #         # TODO is this correct? array does not seem to have NHWC format
    #         for i, dimLen in enumerate(reversed(t.shape)):
    #             packingRunLength *= dimLen
    #             if i + 1 >= packedMinorDims:
    #                 break

    #         # Pack tensor data.
    #         packedData = []
    #         buf = 0
    #         bitsInContainer = 0
    #         mask = (1 << bitsPerItem) - 1
    #         for i, d in enumerate(self.modelT.buffers[t.buffer].data):
    #             d = max(minVal, min(d, maxVal))

    #             buf |= (d & mask) << bitsInContainer
    #             bitsInContainer += bitsPerItem
    #             # Flush full or last container.
    #             if (bitsInContainer + bitsPerItem > containerBits or
    #                     i % packingRunLength == packingRunLength - 1):
    #                 fmtLookup = {8: "B", 16: "H", 32: "I"}
    #                 packedData += struct.pack(fmtLookup[containerBits], buf)

    #                 bitsInContainer = 0
    #                 buf = 0

    #         assert bitsInContainer == 0, "leftover data"

    #         if len(packedData) > len(self.modelT.buffers[t.buffer].data):
    #             print("Warning: Packing increased the size of tensor:", t.name)

    #         self.modelT.buffers[t.buffer].data = [c for c in packedData]

    #     tensorsToOps = {}

    #     for g in self.modelT.subgraphs:
    #         for op in g.operators:
    #             for tIndexList in [op.inputs, op.outputs, op.intermediates]:
    #                 if type(tIndexList) == type(None):
    #                     continue
    #                 for i in tIndexList:
    #                     tensorsToOps.setdefault(i, []).append(op)

    #         # Get candidates for packing.
    #         for tIndex, t in enumerate(g.tensors):
    #             # TODO more types could be supported
    #             if t.type not in [TType.UINT8, TType.INT8]:
    #                 continue

    #             if not t.quantization:
    #                 continue

    #             # Skip if tensor uses custom quantization.
    #             if t.quantization.details:
    #                 continue

    #             if t.buffer == 0 or type(self.modelT.buffers[t.buffer].data) == type(None):
    #                 continue

    #             # Check if all usages of this tensor are okay with packing.
    #             packable = True
    #             for op in tensorsToOps[tIndex]:
    #                 opCode = self.modelT.operatorCodes[op.opcodeIndex].builtinCode
    #                 if opCode not in [OpType.FULLY_CONNECTED, OpType.CONV_2D, OpType.DEPTHWISE_CONV_2D]:
    #                     packable = False

    #                 # TODO detect inconsistent packed minor dims

    #             if not packable:
    #                 continue

    #             packTensor(t, m)

    def mergeModels(models):
        """Merge supplied models to a larger one (if compatible)"""
        raise NotImplementedError

    def getListofOps(self):
        """Returns a list of up incides in the model"""
        return self.modelT.subgraphs[0].operators

    def dropOps(self, indices: list):
        """Remove ops in list of indices from the model"""
        g = self.modelT.subgraphs[0]
        ops = g.operators

        for idx in indices:
            assert idx >= 0 and idx < len(ops)
            # Unfortunately ops can not be dropped completely, we just init a blank op instead
            self.modelT.subgraphs[0].operators[idx] = OperatorT()

    def dropUnusedData(self):
        """Remove all data not related to any operators in the model"""
        g = self.modelT.subgraphs[0]
        ops = g.operators

        used = []

        for op in ops:
            if op is None:
                continue
            if op.inputs is not None:
                used.extend(op.inputs)
            if op.outputs is not None:
                used.extend(op.outputs)
            if op.intermediates is not None:
                used.extend(op.intermediates)

        if self.modelT.metadata:
            for metadata in self.modelT.metadata:
                used.append(metadata.buffer)

        used = list(dict.fromkeys(used))
        not_used = [i for i in range(0, len(g.tensors)) if i not in used]

        for idx in not_used:
            buf = g.tensors[idx].buffer
            self.modelT.buffers[buf].data = None

    def dropData(self):
        """Remove all buffers from the model to reduce its size"""
        for i in range(len(self.modelT.buffers)):
            self.modelT.buffers[i].data = None

    def dropStrings(self):
        """
        Remove all names and descriptions from the moel to reduce its size

        Inspired by `strip_strings(model)` in tensorflow/tensorflow/lite/tools/flatbuffer_utils.py
        """
        self.modelT.description = None
        for subgraph in self.modelT.subgraphs:
            subgraph.name = None
            for tensor in subgraph.tensors:
                tensor.name = None
        if self.modelT.metadata:
            for metadata in self.modelT.metadata:
                metadata.name = None
        self.modelT.signatureDefs = None

    def dropMetadata(self):
        """Get rid of metadata buffers ain the model"""
        bufs = []
        if self.modelT.metadata:
            for metadata in self.modelT.metadata:
                bufs.append(metadata.buffer)
        self.modelT.metadata = []
        for buf in bufs:
            self.modelT.buffers[buf].data = None

    def extractOps(self, idx):
        """Build new model based in the consecutive list of operator indices"""
        # tensorsToOps = {}
        g = self.modelT.subgraphs[0]  # Only supports one subgraph at the moment
        op = g.operators[idx]

        buffers_new = []
        tensors_new = []
        inputs_new = []
        outputs_new = []
        operatorCodes_new = []
        intermediates_new = None  # TODO: unimplemented
        handled_tensor_ids = []
        for i, t in enumerate(op.inputs):
            handled_tensor_ids.append(t)
            tensor_new = g.tensors[t]
            buffer_new = self.modelT.buffers[tensor_new.buffer]
            buffers_new.append(buffer_new)
            tensor_new.buffer = i + 1
            tensors_new.append(tensor_new)
            inputs_new.append(i)
        for i, t in enumerate(op.outputs):
            if t in handled_tensor_ids:
                continue
            handled_tensor_ids.append(t)
            tensor_new = g.tensors[t]
            buffer_new = self.modelT.buffers[tensor_new.buffer]
            buffers_new.append(buffer_new)
            tensor_new.buffer = len(inputs_new) + i + 1
            tensors_new.append(tensor_new)
            outputs_new.append(len(inputs_new) + i)
        operatorCodes_new.append(self.modelT.operatorCodes[op.opcodeIndex])
        self.modelT.operatorCodes = operatorCodes_new
        # TODO: Handle metadata buffer properly
        self.modelT.buffers = [self.modelT.buffers[0]] + buffers_new + [self.modelT.buffers[-1]]
        g.tensors = tensors_new
        op.inputs = np.array(inputs_new, dtype=op.inputs.dtype)
        op.outputs = np.array(outputs_new, dtype=op.outputs.dtype)
        g_inputs_new = [inp for inp in op.inputs if self.modelT.buffers[inp + 1].data is None]
        g_outputs_new = op.outputs
        g.inputs = np.array(g_inputs_new, dtype=g.inputs.dtype)
        g.outputs = np.array(g_outputs_new, dtype=g.outputs.dtype)
        op.intermediates = intermediates_new
        op.opcodeIndex = 0
        if self.modelT.metadata:
            self.modelT.metadata[0].buffer = len(self.modelT.buffers) - 1
        self.modelT.subgraphs[0].operators = [op]
        return self.modelT

    # Print whole model for debugging.
    def printModel(m):
        import jsonpickle
        import yaml

        s = jsonpickle.encode(m)
        print(yaml.dump(yaml.load(s), indent=2))


def main():
    parser = argparse.ArgumentParser(description="TfLite Flatbuffer Rewriter")
    parser.add_argument("model", metavar="MODEL", type=str, help="Flatbuffer file")
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default=os.path.join(os.getcwd(), "output.tflite"),
        help="Output flatbuffer file (default: %(default)s)",
    )
    parser.add_argument(
        "--print", dest="print_model", action="store_true", help="Print resulting model as YAML (default: %(default)s)"
    )
    parser.add_argument("--noop", "-n", action="store_true", help="Skip any transformations (default: %(default)s)")
    parser.add_argument("--pack", "-p", action="store_true", help="Apply packing (default: %(default)s)")
    parser.add_argument(
        "--drop",
        "-d",
        metavar="DROP_LIST",
        type=str,
        default="[]",
        help="Comma-seperated list of node idx to remove from the model, does not alter layout (default: %(default)s)",
    )
    parser.add_argument(
        "--keep",
        "-k",
        metavar="KEEP_LIST",
        type=str,
        default="[]",
        help="Comma-seperated list of consecutive node idx to keep in the model, close gaps (default: %(default)s)",
    )
    parser.add_argument(
        "--trim", dest="trim", action="store_true", help="Drop data/weights completely (default: %(default)s)"
    )
    parser.add_argument(
        "--remove-meta",
        "-r",
        dest="remove_meta",
        action="store_true",
        help="Remove all names and strings from the model (default: %(default)s)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode (default: %(default)s)")
    parser.add_argument("--count-layers", action="store_true", help="Count number of layers (default: %(default)s)")
    args = parser.parse_args()
    # if args.verbose:
    #     logging.basicConfig(level=logging.DEBUG)
    #  TODO: fix this

    assert args.model is not None
    rewriter = TfLiteRewrite(args.model)

    if not args.noop:
        # if args.pack:
        #     rewriter.applyPacking()
        if args.drop != "[]":
            drop_indices = list(map(int, list(filter(None, args.drop.split(",")))))
        else:
            drop_indices = []
        if args.keep != "[]":
            keep_indices = list(map(int, list(filter(None, args.keep.split(",")))))
        else:
            keep_indices = []
        if len(drop_indices) > 0 and len(keep_indices) > 0:
            raise RuntimeError("--drop and --keep can not be used together.")
        elif len(drop_indices) > 0:
            rewriter.dropOps(drop_indices)
            rewriter.dropUnusedData()
        elif len(keep_indices) > 1:
            if sorted(keep_indices) != list(range(min(keep_indices), max(keep_indices) + 1)):
                raise RuntimeError("{} is not a list of consecutive indices.".format(keep_indices))
            raise NotImplementedError
        elif len(keep_indices) == 1:
            rewriter.extractOps(keep_indices[0])
        if args.trim:
            rewriter.dropData()
        if args.remove_meta:
            rewriter.dropMetadata()
            rewriter.dropStrings()

    if args.print_model:
        rewriter.printModel()

    if args.count_layers:
        count = len(rewriter.getListofOps())
        print(f"Found {count} layers.")

    if not args.noop:
        rewriter.saveModelToFile(args.out)


if __name__ == "__main__":
    main()
