from pathlib import Path
import allin1


def main():
    audio_path = Path("/Users/esther/Desktop/New project/songs/周深-生活总该迎着光亮.mp3")

    result = allin1.analyze(str(audio_path))

    print(result)


if __name__ == "__main__":
    main()