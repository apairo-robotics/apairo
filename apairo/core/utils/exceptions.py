class FileExtensionError(Exception):
    pass


class KeysEmptyError(KeyError):
    pass


class KeysDuplicateError(KeyError):
    pass


class EmptyLoaderError(KeyError):
    pass


# Deprecated aliases — these were raised as errors despite the Warning suffix.
KeysEmptyWarning = KeysEmptyError
KeysDuplicateWarning = KeysDuplicateError
EmptyLoaderWarning = EmptyLoaderError
