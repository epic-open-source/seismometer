from IPython.display import HTML, display


def hide_disconnected_widgets(func):
    def wrapper(*args, **kwargs):
        display(HTML("<style> .jupyter-widgets-disconnected { display: none !important }</style>"))
        return func(*args, **kwargs)

    return wrapper
