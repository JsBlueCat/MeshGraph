from .base_options import base_options


class test_options(base_options):
    def initialize(self):
        base_options.initialize(self)
        self.parser.add_argument(
            '--last_epoch', default='latest', help='which epoch to load?'
        )
        self.is_train = False
