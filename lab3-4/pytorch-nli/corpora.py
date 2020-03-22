import torchtext
import torchtext.data as data

class StanfordNLI(data.TabularDataset):

    dirname = 'snli_1.0'
    name = 'snli'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits(cls, text_field, label_field, id_field, parse_field=None, root='.data',
               train='snli_1.0_train.jsonl', validation='snli_1.0_dev.jsonl',
               test='snli_1.0_test.jsonl'):

        path = cls.download(root)

        if parse_field is None:
            return super(StanfordNLI, cls).splits(
                path, root, train, validation, test,
                format='json', fields={'sentence1': ('premise', text_field),
                                       'sentence2': ('hypothesis', text_field),
                                       'gold_label': ('label', label_field),
                                       'pairID' : ('pair_id', id_field)},
                filter_pred=lambda ex: ex.label != '-')
        return super(StanfordNLI, cls).splits(
            path, root, train, validation, test,
            format='json', fields={'sentence1_binary_parse':
                                   [('premise', text_field),
                                    ('premise_transitions', parse_field)],
                                   'sentence2_binary_parse':
                                   [('hypothesis', text_field),
                                    ('hypothesis_transitions', parse_field)],
                                   'gold_label': ('label', label_field),
                                   'pairID' : ('pair_id', id_field)},
            filter_pred=lambda ex: ex.label != '-')


class MultiNLI(data.TabularDataset):

    dirname = 'multinli_1.0'
    name = 'multinli'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits_matched(cls, text_field, label_field, id_field, parse_field=None, genre_field=None, root='.data',
               train='multinli_1.0_train.jsonl', validation='multinli_1.0_dev_matched.jsonl',
               test='multinli_1.0_dev_matched.jsonl'):

        path = cls.download(root)

        if parse_field is None:
            return super(MultiNLI, cls).splits(
                path, root, train, validation, test,
                format='json', fields={'sentence1': ('premise', text_field),
                                       'sentence2': ('hypothesis', text_field),
                                       'gold_label': ('label', label_field),
                                       'pairID' : ('pair_id', id_field)},
                filter_pred=lambda ex: ex.label != '-')
        return super(MultiNLI, cls).splits(
            path, root, train, validation, test,
            format='json', fields={'sentence1_binary_parse':
                                   [('premise', text_field),
                                    ('premise_transitions', parse_field)],
                                   'sentence2_binary_parse':
                                   [('hypothesis', text_field),
                                    ('hypothesis_transitions', parse_field)],
                                   'gold_label': ('label', label_field),
                                   'pairID' : ('pair_id', id_field),
                                   'genre': ('genre', genre_field)},
            filter_pred=lambda ex: ex.label != '-')

    @classmethod
    def splits_mismatched(cls, text_field, label_field, id_field, parse_field=None, genre_field=None, root='.data',
               train='multinli_1.0_train.jsonl', validation='multinli_1.0_dev_mismatched.jsonl',
               test='multinli_1.0_dev_mismatched.jsonl'):

        path = cls.download(root)

        if parse_field is None:
            return super(MultiNLI, cls).splits(
                path, root, train, validation, test,
                format='json', fields={'sentence1': ('premise', text_field),
                                       'sentence2': ('hypothesis', text_field),
                                       'gold_label': ('label', label_field),
                                       'pairID' : ('pair_id', id_field)},
                filter_pred=lambda ex: ex.label != '-')
        return super(MultiNLI, cls).splits(
            path, root, train, validation, test,
            format='json', fields={'sentence1_binary_parse':
                                   [('premise', text_field),
                                    ('premise_transitions', parse_field)],
                                   'sentence2_binary_parse':
                                   [('hypothesis', text_field),
                                    ('hypothesis_transitions', parse_field)],
                                   'gold_label': ('label', label_field),
                                   'pairID' : ('pair_id', id_field),
                                   'genre': ('genre', genre_field)},
            filter_pred=lambda ex: ex.label != '-')
