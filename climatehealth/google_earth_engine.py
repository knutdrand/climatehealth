import ee


def read_data():
    # Trigger the authentication flow.
    ee.Authenticate()

    # Initialize the library.
    ee.Initialize()
