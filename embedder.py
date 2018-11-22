import math
import os

import numpy as np
import torch
import torch.nn as nn


class V2S(torch.nn.Module):

	def __init__(self,
				 video_dim,
				 latent_dim,
				 word_dim):
		super(V2S, self).__init__()

		self.fc1 = nn.Linear(video_dim, latent_dim)
		self.fc2 = nn.Linear(latent_dim, latent_dim)
		self.fc3 = nn.Linear(latent_dim, word_dim)

	def forward(self, inp):
		out = self.fc1(inp)
		out = self.fc2(out)
		out = self.fc3(out)
		return out


class VideoEncoder(torch.nn.Module):

	def __init__(self,
				 video_dim,
				 latent_dim):
		super(VideoEncoder, self).__init__()

		self.conv1 = nn.Conv1d(video_dim, 1, 30)
		self.max1 = nn.MaxPool1d(4, return_indices=True)
		self.conv2 = nn.Conv1d(1, 1, 10)

	def forward(self, inp):
		out = self.conv1(inp)
		# print(out.shape)
		out, i1 = self.max1(out)
		# print(out.shape)
		out = self.conv2(out)
		return out, i1


class VideoDecoder(torch.nn.Module):

	def __init__(self,
				 video_dim,
				 latent_dim):
		super(VideoDecoder, self).__init__()

		self.deconv1 = nn.ConvTranspose1d(1, 1, 10)
		self.unpool1 = nn.MaxUnpool1d(4)
		self.deconv2 = nn.ConvTranspose1d(1, 1, 30)

	def forward(self, inp, i1):
		out = self.deconv1(inp)
		# print(out.shape)
		out = self.unpool1(out, i1, output_size=torch.Size([1, 1, 995]))
		# print(out.shape)
		out = self.deconv2(out)
		return out


class WordEncoder(torch.nn.Module):

	def __init__(self,
				 word_dim):
		super(WordEncoder, self).__init__()

		self.conv1 = nn.Conv1d(word_dim, 1, 30)
		self.max1 = nn.MaxPool1d(2, return_indices=True)
		self.conv2 = nn.Conv1d(1, 1, 10)

	def forward(self, inp):
		out = self.conv1(inp)
		print(out.shape)
		out, i1 = self.max1(out)
		print(out.shape)
		out = self.conv2(out)
		return out, i1


class WordDecoder(torch.nn.Module):

	def __init__(self,
				 word_dim):
		super(WordDecoder, self).__init__()

		self.deconv1 = nn.ConvTranspose1d(1, 1, 10)
		self.unpool1 = nn.MaxUnpool1d(2)
		self.deconv2 = nn.ConvTranspose1d(1, 1, 30)

	def forward(self, inp, i1):
		out = self.deconv1(inp)
		print(out.shape)
		out = self.unpool1(out, i1, output_size=torch.Size([1, 1, 271]))
		print(out.shape)
		out = self.deconv2(out)
		return out


class VideoDiscriminator(torch.nn.Module):

	def __init__(self,
				video_dim):
		super(VideoDiscriminator, self).__init__()


class TextDiscriminator(torch.nn.Module):

	def __init__(self,
				 word_dim):
		super(TextDiscriminator, self).__init__()


class LatentDiscriminator(torch.nn.Module):

	def __init__(self,
				 latent_dim):
		super(LatentDiscriminator, self).__init__()


def loss_embedding(enc_v, dec_v, enc_t, dec_t, v, t):
	alpha1, alpha2, alpha3 = 0.1, 0.1, 0.1
	pdist = nn.PairwiseDistance(p=2)
	l_recons = pdist(v, dec_v(enc_v(v))) + pdist(t, dec_t(enc_t(t)))
	l_joint = pdist(enc_v(v), enc_t(t))
	l_cross = pdist(dec_t(enc_v(v)), t) + pdist(dec_v(enc_t(t)), v)
	l_cycle = pdist(t, dec_t(enc_v(dec_v(enc_t(t))))) + pdist(v, dec_v(enc_t(dec_t(enc_v(v)))))
	loss = l_recons + alpha1*l_joint + alpha2*l_cross + alpha3*l_cycle
	return loss


def loss_videodisc(enc_t, dec_v, t, v, disc_v):
	loss = nn.BCELoss()

	pred_real = disc_v(v)
	error_real = loss(pred_real, one_target)
	error_real.backward()

	pred_fake = disc_v(dec_v(enc_t(t)))
	error_fake = loss(pred_fake, zero_target)
	error_fake.backward()


def loss_textdisc(encv, dec_t, t, v, disc_t):
	pass


def loss_latentdisc(enc_t, enc_v, t, v, disc_l):
	loss = nn.BCELoss()

	pred_v = enc_v(v)
	error_v = loss(pred_v, one_target)
	error_v.backward()

	pred_t = enc_t(t)
	error_t = loss(pred_t, zero_target)
	error_t.backward()


def loss_videogen(enc_t, dec_v, t, disc_v):
	loss = nn.BCELoss
	pred = disc_v(dec_v(enc_t(t)))
	err = loss(pred, ones_target)
	err.backward()


def loss_textgen(enc_v, dec_t, v, disc_t):
	pass


def loss_latentgen(enc_t, enc_v, t, v, disc_lat):
	loss = nn.BCELoss()

	pred_v = disc_lat(enc_v(v))
	error_v = loss(pred_v, one_target)
	error_v.backward()

	pred_t = disc_lat(enc_t(t))
	error_t = loss(pred_t, zero_target)
	error_t.backward()




