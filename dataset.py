import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, images, tags, labels, test=None):
        self.test = test
        if test is None:
            train_images = images[opt.query_size: opt.query_size + opt.training_size]
            train_tags = tags[opt.query_size: opt.query_size + opt.training_size]
            train_labels = labels[opt.query_size: opt.query_size + opt.training_size]
            self.images, self.tags, self.labels = train_images, train_tags, train_labels
        else:
            self.query_labels = labels[0: opt.query_size]
            self.db_labels = labels[opt.query_size: opt.query_size + opt.db_size]
            if test == 'image.query':
                self.images = images[0: opt.query_size]
            elif test == 'image.db':
                self.images = images[opt.query_size: opt.query_size + opt.db_size]
            elif test == 'text.query':
                self.tags = tags[0: opt.query_size]
            elif test == 'text.db':
                self.tags = tags[opt.query_size: opt.query_size + opt.db_size]

    def __getitem__(self, index):
        if self.test is None:
            return (
                index,
                torch.from_numpy(self.images[index].astype('float32')),
                torch.from_numpy(self.tags[index].astype('float32')),
                torch.from_numpy(self.labels[index].astype('float32'))
            )
        elif self.test.startswith('image'):
            return torch.from_numpy(self.images[index].astype('float32'))
        elif self.test.startswith('text'):
            return torch.from_numpy(self.tags[index].astype('float32'))

    def __len__(self):
        if self.test is None:
            return len(self.images)
        elif self.test.startswith('image'):
            return len(self.images)
        elif self.test.startswith('text'):
            return len(self.tags)

    def get_labels(self):
        if self.test is None:
            return torch.from_numpy(self.labels.astype('float32'))
        else:
            return (
                torch.from_numpy(self.query_labels.astype('float32')),
                torch.from_numpy(self.db_labels.astype('float32'))
            )