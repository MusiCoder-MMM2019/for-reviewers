import tensorflow as tf
import os
import utils
import texture_transfer_tester
import argparse
import time

def parse_args():
    desc = 'taken from Fast texture Transfer for Image'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--texture_model', type=str, default='models/water.ckpt', help='location for model file (*.ckpt)',
                        required=True)

    parser.add_argument('--content', type=str, default='../temp/content.jpg',
                        help='File path of content image (notation in the paper : x)', required=True)

    parser.add_argument('--output', type=str, default='../temp/transformed.jpg',
                        help='File path of output image (notation in the paper : y_c)', required=True)

    parser.add_argument('--max_size', type=int, default=None, help='The maximum width or height of input images')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    try:
        assert os.path.exists(args.texture_model + '.index') and os.path.exists(args.texture_model + '.meta') and os.path.exists(
            args.texture_model + '.data-00000-of-00001')
    except:
        print('There is no %s'%args.texture_model)
        print('Tensorflow r0.12 requires 3 files related to *.ckpt')
        print('If you want to restore any models generated from old tensorflow versions, this assert might be ignored')
        return None

    try:
        assert os.path.exists(args.content)
    except:
        print('There is no %s' % args.content)
        return None

    try:
        if args.max_size is not None:
            assert args.max_size > 0
    except:
        print('The maximum width or height of input image must be positive')
        return None

    dirname = os.path.dirname(args.output)
    try:
        if len(dirname) > 0:
            os.stat(dirname)
    except:
        os.mkdir(dirname)

    return args

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    args = parse_args()
    if args is None:
        exit()


    content_image = utils.load_image(args.content, max_size=args.max_size)


    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True 
    sess = tf.Session(config=soft_config)

    transformer = texture_transfer_tester.textureTransferTester(session=sess,
                                                            model_path=args.texture_model,
                                                            content_image=content_image,
                                                            )

    start_time = time.time()
    output_image = transformer.test()
    end_time = time.time()


    utils.save_image(output_image, args.output)


    shape = content_image.shape
    print('Execution DONE in: %f msec' % (1000.*float(end_time - start_time)/60))

if __name__ == '__main__':
    main()
