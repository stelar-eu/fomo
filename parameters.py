class Parameters:
    # Required attributes (to be set)
    input_path: str = None
    window: int = None

    # Optional attributes
    budget: int = 100
    metric: str = "euclidean"
    n_streams: int = None
    duration: int = None
    warmup: int = 0
    selection_strategy: str = 'odac'
    prio_strategy: str = 'rmse'
    tau: float = 1
    index: bool = False
    header: bool = False

    @staticmethod
    def check():
        """
        Check if the parameters are set correctly
        """
        p = Parameters()

        assert p.input_path is not None, "No input path provided"
        assert p.window is not None, "No window size provided"
        assert p.selection_strategy in ['odac', 'singleton'], f"Invalid selection strategy: {p.selection_strategy}"
        assert p.prio_strategy in ['rmse', 'random'], f"Invalid prioritization strategy: {p.prio_strategy}"
