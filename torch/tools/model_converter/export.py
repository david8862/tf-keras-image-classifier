#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os, sys, argparse
import torch
#from torchsummary import summary

# add root path of model definition here,
# to make sure that we can load .pth model file with torch.load()
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))


def model_export(model_path, model_input_shape, op_set, output_path, batch_size):
    # Input
    if batch_size == -1:
        img = torch.zeros((1, 3, *model_input_shape))
    else:
        img = torch.zeros((batch_size, 3, *model_input_shape))

    # Load PyTorch model
    model = torch.load(model_path, map_location=torch.device('cpu')).float()
    model.eval()
    y = model(img)  # dry run

    # Strip model file name
    model_basename = os.path.basename(model_path).split('.')
    model_basename = '.'.join(model_basename[:-1])
    os.makedirs(output_path, exist_ok=True)

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        export_file = os.path.join(output_path, model_basename+'.torchscript.pt')

        ts = torch.jit.trace(model, img)
        ts.save(export_file)
        print('TorchScript export success, saved as %s' % export_file)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        export_file = os.path.join(output_path, model_basename+'.onnx')

        if batch_size == -1:
            # dump dynamic batch-size onnx model
            torch.onnx.export(model, img, export_file, verbose=False, opset_version=op_set, input_names=['image_input'], output_names=['scores'],
                              dynamic_axes={"image_input": {0: "batch_size"}, "scores": {0: "batch_size"}})

        else:
            # dump fix batch-size onnx model
            torch.onnx.export(model, img, export_file, verbose=False, opset_version=op_set, input_names=['image_input'], output_names=['scores'])

        # Checks
        onnx_model = onnx.load(export_file)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % export_file)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # CoreML export
    #try:
        #import coremltools as ct

        #print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        ## convert model from torchscript and apply pixel scaling as per detect.py
        #model = ct.convert(ts, inputs=[ct.ImageType(name='images', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        #export_file = os.path.join(output_path, model_basename+'.mlmodel')
        #model.save(export_file)
        #print('CoreML export success, saved as %s' % export_file)
    #except Exception as e:
        #print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='model file to export')
    parser.add_argument('--model_input_shape', type=str, required=False, help='model input image shape as <height>x<width>, default=%(default)s', default='224x224')
    parser.add_argument('--op_set', type=int, required=False, help='onnx op set, default=%(default)s', default=14)
    parser.add_argument('--batch_size', type=int, required=False, help="batch size for inference, default=%(default)s", default=-1)
    parser.add_argument('--output_path', type=str, required=True, help='output path for exported model')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    model_export(args.model_path, args.model_input_shape, args.op_set, args.output_path, args.batch_size)


if __name__ == "__main__":
    main()



