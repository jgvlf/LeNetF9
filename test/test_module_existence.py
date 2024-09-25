import lenetf9


def test_module_existence() -> None:
    if not lenetf9:
        msg: tuple[str, str] = (
            'Unrefence "lenetf9" package. ',
            "Do You install this package in your virtual environment?",
        )
        raise ModuleNotFoundError(msg)
