"""Console script for climatehealth."""
# todo


import typer


class Examples:
    @classmethod
    def laos(cls):


analyses = {'laos_example': Examples.laos}

def main(analysis_name: str):
    '''
    Simple function

    >>> main()
    '''
    if analysis_name not in analyses:
        raise ValueError(f'Unknown analysis {analysis_name}, available analyses are {list(analyses)}')
    analyses[analysis_name]()


if __name__ == "__main__":
    typer.run(main)
