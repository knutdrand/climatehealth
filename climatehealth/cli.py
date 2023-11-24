"""Console script for climatehealth."""
# todo


import typer


class Examples:
    @classmethod
    def laos(cls, base_folder: str):
        from .examples.laos_example import main
        main(base_folder + '/example_data/10yeardengudata.csv')


analyses = {'laos_example': Examples.laos}


def main(analysis_name: str='laos_example', base_folder : str = 'example_data'):
    '''
    Simple function

    >>> main()
    '''
    if analysis_name not in analyses:
        raise ValueError(f'Unknown analysis {analysis_name}, available analyses are {list(analyses)}')
    analyses[analysis_name](base_folder)


if __name__ == "__main__":
    typer.run(main)
