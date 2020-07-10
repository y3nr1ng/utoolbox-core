from click.testing import CliRunner
from utoolbox.cli.aszarr import aszarr


def test_aszarr_latticescope():
    runner = CliRunner()
    runner.invoke(
        aszarr,
        [
            "-vvv",
            "-r",
            "yxz",
            "-f",
            "x",
            "c:\\users\\andy\\desktop\\utoolbox\\utoolbox-core\\workspace\\data\\20200704_kidney_demo-2_CamA",
        ],
    )
    print(runner)


if __name__ == "__main__":
    test_aszarr_latticescope()
