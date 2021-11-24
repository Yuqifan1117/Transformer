import os
from torchtext.datasets import TranslationDataset

class TextListData(TranslationDataset):
    """TextData"""

    urls = None
    name = 'text_list'
    dirname = None

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='val', test='test', **kwargs):

        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(TextListData, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)