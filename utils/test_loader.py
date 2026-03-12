from data_loader import get_data_loaders

if __name__ == '__main__':
    train_loader, val_loader, test_loader, idx_to_class = get_data_loaders(
        '../data', batch_size=4, train_augment=False
    )
    print("类别映射:", idx_to_class)
    for images, labels in train_loader:
        print(images.shape, labels)
        break

"""
原本的是：
from data_loader import get_data_loaders
if __name__ == '__main__':
    train_loader, val_loader, test_loader, idx_to_class = get_data_loaders(
        '../data', batch_size=4, train_augment=False
    )
    print("类别映射:", idx_to_class)
    for images, labels in train_loader:
        print(images.shape, labels)
        break

if __name__ == '__main__': 确保只有在直接运行该脚本时才执行里面的代码，而不会在被其他模块导入时执行。
同时，由于我们在 Windows 上，可以暂时将 num_workers 设为 0 来避免多进程问题，但更好的做法是保留 num_workers=2 并加上保护，
这样在 Linux/Mac 上也能充分利用多核。我们已经在 get_data_loaders 中设置了 num_workers=2，现在通过主程序保护即可。
"""