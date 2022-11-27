import pygame

image = pygame.image.load(".\\data\\numbers.png")

def render(window:pygame.Surface, number:int, x, y, color=(255,255,255)) -> None:
    number = int(number)
    image.fill(color, special_flags=pygame.BLEND_ADD)
    window.blit(image, (x, y), [16*number, 0, 16, 16])