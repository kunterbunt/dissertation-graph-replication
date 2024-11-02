from os import path

import gym
import numpy as np

# Adapted from: https://github.com/openai/gym/blob/bc17d9f7e7e4c033cd3674e6331e459cef9f7eb2/gym/envs/classic_control/cartpole.py
class Renderer(gym.Wrapper):
	metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 30}

	def __init__(self, env):
		super().__init__(env)

		self.screen_width = 1600
		self.screen_height = 900
		self.channel_box_size = 16

		self.screen = None
		self.clock = None
		self.isopen = False

		self.history = []
		self.info = None
		self.channel_surf = None

		self.map_image = None
		self.aircraft_image = None
		self.aircraft_tagged_image = None
		self.ground_station_image = None

	def reset(self, **kwargs):
		obs = self.env.reset(**kwargs)

		self.history = []
		self.info = None
		if self.channel_surf is not None:
			self.channel_surf.fill((128, 128, 128))

		return obs

	def step(self, action):
		obs, reward, done, truncated, info = self.env.step(action)

		if self.screen is not None and self.screen != False:
			import pygame
			from pygame.locals import QUIT

			for event in pygame.event.get():
				if event.type == QUIT:
					done = True

		self.history.append((obs, reward, action))
		self.info = info

		return obs, reward, done, truncated, info

	def render(self, mode="human"):
		import pygame
		from pygame import gfxdraw

		# Initialize renderer
		if self.isopen == False:
			pygame.init()

			self.map_image = pygame.transform.smoothscale(
				pygame.image.load(path.join(path.dirname(__file__), "images", "map.png")),
				(self.screen_width, self.screen_height))
			self.aircraft_image = pygame.transform.smoothscale(
				pygame.image.load(path.join(path.dirname(__file__), "images", "aircraft.png")),
				(int(32), int(32)))
			self.aircraft_tagged_image = pygame.transform.smoothscale(
				pygame.image.load(path.join(path.dirname(__file__), "images", "aircraft_tagged.png")),
				(int(32), int(32)))
			self.ground_station_image = pygame.transform.smoothscale(
				pygame.image.load(path.join(path.dirname(__file__), "images", "ground_station.png")),
				(int(32), int(32)))

			self.channel_surf = pygame.Surface((self.channel_box_size*self.num_freq_channels, self.screen_height))
			self.channel_surf.fill((128, 128, 128))

			self.isopen = True

		# Initialize screen and clock
		if mode == "human":
			if self.screen is None:
				try:
					pygame.display.init()
					pygame.display.set_caption("TUHH I³ Project: Machine Learning for Communications in Aviation")
					self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
				except pygame.error as e:
					print("Cannot initialize pygame display:", e)
					self.screen = False
			if self.clock is None:
				self.clock = pygame.time.Clock()

		# Initialize render surface
		self.surf = pygame.Surface((self.screen_width, self.screen_height))

		# Draw map
		self.surf.blit(self.map_image, (0, 0))

		# Draw markers
		if self.info is not None:
			# Draw DME ground stations
			if "ground_stations" in self.info:
				for ground_station in self.info["ground_stations"]:
					pos_x = ground_station["pos_x"]
					pos_y = ground_station["pos_y"]
					ground_station_x = self.screen_width*pos_x - self.ground_station_image.get_width()/2
					ground_station_y = self.screen_height*pos_y - self.ground_station_image.get_height()/2
					self.surf.blit(self.ground_station_image, (ground_station_x, ground_station_y))

			# Draw aircraft
			if "aircraft" in self.info:
				for aircraft_id, aircraft in enumerate(self.info["aircraft"]):
					pos_x = aircraft["pos_x"]
					pos_y = aircraft["pos_y"]
					rot = aircraft["rot"]
					aircraft_image_rotated = pygame.transform.rotate(self.aircraft_image if aircraft_id > 0 else self.aircraft_tagged_image, rot-90)
					aircraft_x = self.screen_width*pos_x - aircraft_image_rotated.get_width()/2
					aircraft_y = self.screen_height*pos_y - aircraft_image_rotated.get_height()/2
					self.surf.blit(aircraft_image_rotated, (aircraft_x, aircraft_y))

		# Draw channels
		for (observation, _, action) in self.history:
			self.channel_surf.scroll(0, self.channel_box_size)

			for channel, state in enumerate(observation):
				if state == 1:
					color = (255, 255, 255)
				elif state == -1:
					color = (0, 0, 0)
				else:
					color = (128, 128, 128)
				gfxdraw.box(self.channel_surf, pygame.Rect(self.channel_box_size*channel, 0, self.channel_box_size, self.channel_box_size), color)

			gfxdraw.rectangle(self.channel_surf, pygame.Rect(self.channel_box_size*action, 0, self.channel_box_size, self.channel_box_size), (255, 0, 0))
		self.history.clear()

		self.surf.blit(self.channel_surf, (0, 0))

		# Output
		if mode == "human":
			if self.screen is not None and self.screen != False:
				self.screen.blit(self.surf, (0, 0))
				pygame.event.pump()
				self.clock.tick(self.metadata["render_fps"])
				pygame.display.flip()
		elif mode == "rgb_array":
			return np.transpose(
				np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
			)

		return self.isopen

	def close(self):
		if self.isopen == True:
			import pygame

			if self.screen is not None and self.screen != False:
				pygame.display.quit()

			pygame.quit()

			self.screen = None
			self.clock = None
			self.isopen = False
