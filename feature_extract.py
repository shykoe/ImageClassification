from __future__ import print_function, division
import os
import argparse
import sys
import torch
import pretrainedmodels
import pretrainedmodels.utils
import pretrainedmodels.datasets as datasets
import h5py

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))
parser = argparse.ArgumentParser(
    description='ImageNet Feature Extraction',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_outputs', default='./outputs.h5', type=str, help='')
parser.add_argument('--dir_datasets', default='/tmp/datasets', type=str, help='')
parser.add_argument('-b', '--batch_size', default=30, type=float, help='')
parser.add_argument('-a', '--arch', default='alexnet', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--cuda', const=True, nargs='?', type=bool, help='')
args = parser.parse_args()
def main():
    pass

if __name__ == '__main__':

    model = pretrainedmodels.__dict__[args.arch]()
    features_size = model.last_linear.in_features
    model.last_linear = pretrainedmodels.utils.Identity() 
    scale = 0.875
    val_tf = pretrainedmodels.utils.TransformImage(
        model,
        scale=scale,
        preserve_aspect_ratio=True
    )
    dataset = datasets.ExtractionDatasetFolder(args.dir_datasets,
                                                 val_tf)
    total = len(dataset)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    if os.path.exists(args.dir_outputs):
        raise RuntimeError('{} file exits'.format(args.dir_outputs))
    df = h5py.File(args.dir_outputs,'w')
    dst = df.create_dataset("features",(total, features_size),dtype='float32')
    with torch.no_grad():
        for ind, (img, label) in enumerate(val_loader):
            out = model(img)
            dst[ind*args.batch_size:(ind+1)*args.batch_size, :] = out
            print('{}/{}'.format(min((ind+1)*args.batch_size,total), total))

    df.close()
