from ultralytics import YOLO
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cpu', help='cpu or gpu mode')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--data', type=str, default='data/data.yaml', help='data config path')
    # parser.add_argument('--model', type=str, default='models/yolov9m-custom.yaml', help='model config path')
    # parser.add_argument('--weights', type=str, default='weights/pretrained.pt', help='pretrained weights path')
    args = parser.parse_args()

    # Device configuration
    device = 'cuda:0' if args.mode == 'gpu' else 'cpu'
    print(f"Using device: {device}")

    # Load model, if available else train from scratch
    model = YOLO('yolov9m.pt')
    # model = YOLO(args.model)  # build from YAML
    # model = YOLO(args.weights)  # load a pretrained model

    # Training configuration
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'device': device,
        'imgsz': 640,
        'workers': 2 if args.mode == 'gpu' else 2,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
    }

    # Train the model
    results = model.train(**train_args)

if __name__ == '__main__':
    main()