"""Local alert-sound playback."""

import pygame


class SoundManager:
    """Load, play, stop, and close local alert sounds."""

    def __init__(self, sound_files):
        self.sounds = {}
        self.mixer_ready = False

        try:
            pygame.mixer.init()
            self.mixer_ready = True
        except pygame.error as error:
            print(f"Could not start audio: {error}")
            return

        for name, path in sound_files.items():
            if not path.is_file():
                print(f"Sound file not found: {path}")
                continue

            try:
                self.sounds[name] = pygame.mixer.Sound(str(path))
            except pygame.error as error:
                print(f"Could not load {name}: {error}")

    def play(self, name):
        sound = self.sounds.get(name)

        if sound is not None:
            sound.stop()
            sound.play()

    def stop(self, name):
        sound = self.sounds.get(name)

        if sound is not None:
            sound.stop()

    def close(self):
        for sound in self.sounds.values():
            sound.stop()

        if self.mixer_ready:
            pygame.mixer.quit()


def update_away_sound(
    sound_manager,
    was_away_alert_active,
    away_alert_is_active,
    priority_alert_is_active,
):
    """Play the moved-away sound once when its alert begins."""
    if (
        away_alert_is_active
        and not was_away_alert_active
        and not priority_alert_is_active
    ):
        sound_manager.play("moved_away")

    if was_away_alert_active and not away_alert_is_active:
        sound_manager.stop("moved_away")
