def __getattr__(name):
    if name == "__version__":
        return "0.1.0"
    raise AttributeError
