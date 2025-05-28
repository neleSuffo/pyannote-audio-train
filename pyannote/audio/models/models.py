#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
# Juan Manuel Coria

from typing import Optional


import torch
import torch.nn as nn
import logging
import numpy as np
import librosa
from .sincnet import SincNet
from .tdnn import XVectorNet
from .pooling import TemporalPooling
from pyannote.audio.train.model import Model
from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import RESOLUTION_CHUNK
from pyannote.audio.train.model import RESOLUTION_FRAME

logger = logging.getLogger(__name__)

class RNN(nn.Module):
    """Recurrent layers

    Parameters
    ----------
    n_features : `int`
        Input feature shape.
    unit : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    hidden_size : `int`, optional
        Number of features in the hidden state h. Defaults to 16.
    num_layers : `int`, optional
        Number of recurrent layers. Defaults to 1.
    bias : `boolean`, optional
        If False, then the layer does not use bias weights. Defaults to True.
    dropout : `float`, optional
        If non-zero, introduces a Dropout layer on the outputs of each layer
        except the last layer, with dropout probability equal to dropout.
        Defaults to 0.
    bidirectional : `boolean`, optional
        If True, becomes a bidirectional RNN. Defaults to False.
    concatenate : `boolean`, optional
        Concatenate output of each layer instead of using only the last one
        (which is the default behavior).
    pool : {'sum', 'max', 'last', 'x-vector'}, optional
        Temporal pooling strategy. Defaults to no pooling.
    """

    def __init__(self, n_features, unit='LSTM', hidden_size=16, num_layers=1,
                 bias=True, dropout=0, bidirectional=False, concatenate=False,
                 pool=None):
        super().__init__()

        self.n_features = n_features

        self.unit = unit
        Klass = getattr(nn, self.unit)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.concatenate = concatenate
        self.pool = pool
        self.pool_ = TemporalPooling.create(pool) if pool is not None else None

        if num_layers < 1:
            msg = ('"bidirectional" must be set to False when num_layers < 1')
            if bidirectional:
                raise ValueError(msg)
            msg = ('"concatenate" must be set to False when num_layers < 1')
            if concatenate:
                raise ValueError(msg)
            return

        if self.concatenate:

            self.rnn_ = nn.ModuleList([])
            for i in range(self.num_layers):

                if i > 0:
                    input_dim = self.hidden_size
                    if self.bidirectional:
                        input_dim *= 2
                else:
                    input_dim = self.n_features

                if i + 1 == self.num_layers:
                    dropout = 0
                else:
                    dropout = self.dropout

                rnn = Klass(input_dim, self.hidden_size,
                            num_layers=1, bias=self.bias,
                            batch_first=True, dropout=dropout,
                            bidirectional=self.bidirectional)

                self.rnn_.append(rnn)

        else:
            self.rnn_ = Klass(self.n_features, self.hidden_size,
                              num_layers=self.num_layers, bias=self.bias,
                              batch_first=True, dropout=self.dropout,
                              bidirectional=self.bidirectional)

    def forward(self, features, return_intermediate=False):
        """Apply recurrent layer (and optional temporal pooling)

        Parameters
        ----------
        features : `torch.Tensor`
            Features shaped as (batch_size, n_frames, n_features)
        return_intermediate : `boolean`, optional
            Return intermediate RNN hidden state.

        Returns
        -------
        output : `torch.Tensor`
            TODO. Shape depends on parameters...
        intermediate : `torch.Tensor`
            (num_layers, batch_size, hidden_size * num_directions)
        """

        if self.num_layers < 1:

            if return_intermediate:
                msg = ('"return_intermediate" must be set to False '
                       'when num_layers < 1')
                raise ValueError(msg)

            output = features

        else:

            if return_intermediate:
                num_directions = 2 if self.bidirectional else 1

            if self.concatenate:

                if return_intermediate:
                    msg = (
                        '"return_intermediate" is not supported '
                        'when "concatenate" is True'
                    )
                    raise NotADirectoryError(msg)

                outputs = []

                # apply each layer separately...
                for i, rnn in enumerate(self.rnn_):
                    if i > 0:
                        output, hidden = rnn(output, hidden)
                    else:
                        output, hidden = rnn(features)
                    outputs.append(output)

                # ... and concatenate their output
                output = torch.cat(outputs, dim=2)

            else:
                output, hidden = self.rnn_(features)

                if return_intermediate:
                    if self.unit == 'LSTM':
                        h = hidden[0]
                    elif self.unit == 'GRU':
                        h = hidden

                    # to (num_layers, batch_size, num_directions * hidden_size)
                    h = h.view(
                        self.num_layers, num_directions, -1, self.hidden_size)
                    intermediate = h.transpose(2, 1).contiguous().view(
                        self.num_layers, -1, num_directions * self.hidden_size)

        if self.pool_ is not None:
            output = self.pool_(output)

        if return_intermediate:
            return output, intermediate

        return output

    def dimension():
        doc = "Output features dimension."
        def fget(self):
            if self.num_layers < 1:
                dimension = self.n_features
            else:
                dimension = self.hidden_size

            if self.bidirectional:
                dimension *= 2

            if self.concatenate:
                dimension *= self.num_layers

            if self.pool == 'x-vector':
                dimension *= 2

            return dimension
        return locals()
    dimension = property(**dimension())

    def intermediate_dimension(self, layer):
        if self.num_layers < 1:
            dimension = self.n_features
        else:
            dimension = self.hidden_size

        if self.bidirectional:
            dimension *= 2

        return dimension


class FF(nn.Module):
    """Feedforward layers

    Parameters
    ----------
    n_features : `int`
        Input dimension.
    hidden_size : `list` of `int`, optional
        Linear layers hidden dimensions. Defaults to [16, ].
    """

    def __init__(self, n_features, hidden_size=[16, ]):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size

        self.linear_ = nn.ModuleList([])
        for hidden_size in self.hidden_size:
            linear = nn.Linear(n_features, hidden_size, bias=True)
            self.linear_.append(linear)
            n_features = hidden_size

    def forward(self, features):
        """

        Parameters
        ----------
        features : `torch.Tensor`
            (batch_size, n_samples, n_features) or (batch_size, n_features)

        Returns
        -------
        output : `torch.Tensor`
            (batch_size, n_samples, hidden_size[-1]) or (batch_size, hidden_size[-1])
        """

        output = features
        for linear in self.linear_:
            output = linear(output)
            output = torch.tanh(output)
        return output

    def dimension():
        doc = "Output dimension."
        def fget(self):
            if self.hidden_size:
                return self.hidden_size[-1]
            return self.n_features
        return locals()
    dimension = property(**dimension())


class Embedding(nn.Module):
    """Embedding

    Parameters
    ----------
    n_features : `int`
        Input dimension.
    batch_normalize : `boolean`, optional
        Apply batch normalization. This is more or less equivalent to
        embedding whitening.
    unit_normalize : `boolean`, optional
        Normalize embeddings. Defaults to False.
    """

    def __init__(self, n_features, batch_normalize=False, unit_normalize=False):
        super().__init__()

        self.n_features = n_features
        # batch normalization ~= embeddings whitening.

        self.batch_normalize = batch_normalize
        if self.batch_normalize:
            self.batch_normalize_ = nn.BatchNorm1d(
                n_features, eps=1e-5, momentum=0.1, affine=False)

        self.unit_normalize = unit_normalize

    def forward(self, embedding):

        if self.batch_normalize:
            embedding = self.batch_normalize_(embedding)

        if self.unit_normalize:
            norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
            embedding = embedding / norm

        return embedding

    def dimension():
        doc = "Output dimension."
        def fget(self):
            return self.n_features
        return locals()
    dimension = property(**dimension())

class PyanNetEnhanced(Model):
    """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output

    Parameters
    ----------
    sincnet : `dict`, optional
        SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters. Use {'skip': True} to use handcrafted features
        instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
        becomes [ features -> RNN -> ...].
    rnn : `dict`, optional
        Recurrent network parameters. Defaults to `RNN` default parameters.
    ff : `dict`, optional
        Feed-forward layers parameters. Defaults to `FF` default parameters.
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
    add_rms : `bool`, optional
        Whether to add RMS energy as an additional feature. Defaults to True.
    add_sf : `bool`, optional
        Whether to add spectral flatness as an additional feature. Defaults to True.
    """
    @staticmethod
    def get_alignment(sincnet=None, **kwargs):
        if sincnet is None:
            sincnet = dict()
        if sincnet.get('skip', False):
            return 'center'
        return SincNet.get_alignment(**sincnet)

    supports_packed = False

    @staticmethod
    def get_resolution(sincnet: Optional[dict] = None, rnn: Optional[dict] = None, **kwargs) -> Resolution:
        """Get sliding window used for feature extraction

        Parameters
        ----------
        sincnet : dict, optional
        rnn : dict, optional

        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
            Returns RESOLUTION_CHUNK if model returns one vector per input
            chunk, RESOLUTION_FRAME if model returns one vector per input
            frame, and specific sliding window otherwise.
        """

        if rnn is None:
            rnn = {'pool': None}
        if rnn.get('pool', None) is not None:
            return RESOLUTION_CHUNK
        if sincnet is None:
            sincnet = {'skip': False}
        if sincnet.get('skip', False):
            return RESOLUTION_FRAME
        return SincNet.get_resolution(**sincnet)

    def init(self,
             sincnet: Optional[dict] = None,
             rnn: Optional[dict] = None,
             ff: Optional[dict] = None,
             embedding: Optional[dict] = None,
             add_rms: bool = True,
             add_sf: bool = True,
             add_formants: bool = True,
             add_pitch: bool = True,
             add_mfcc: bool = True):
        """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output

        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters. Use {'skip': True} to use handcrafted features
            instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
            becomes [ features -> RNN -> ...].
        rnn : `dict`, optional
            Recurrent network parameters. Defaults to `RNN` default parameters.
        ff : `dict`, optional
            Feed-forward layers parameters. Defaults to `FF` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        add_rms: bool
            feature to extracrt
        add_sf: bool
            feature to extract
        add_formants: bool
            feature to extract
        add_pitch: bool
            feature to extract
        add_mfcc: bool
            feature to extract
        """
        print("--- PyanNetEnhanced init method CALLED ---", flush=True)
        self.add_rms = add_rms
        self.add_sf = add_sf
        self.add_formants = add_formants
        self.add_pitch = add_pitch
        self.add_mfcc = add_mfcc
        n_features = self.n_features

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet

        if not sincnet.get('skip', False):
            if n_features != 1:
                raise ValueError(f'SincNet only supports mono waveforms. Here, waveform has {n_features} channels.')
            self.sincnet_ = SincNet(**sincnet)
            n_features = self.sincnet_.dimension
            n_features_for_rnn = n_features
            if self.add_rms:
                n_features_for_rnn += 1
            if self.add_sf:
                n_features_for_rnn += 1
            if self.add_formants:
                n_features_for_rnn += 2  # F1, F2
            if self.add_pitch:
                n_features_for_rnn += 1  # Pitch variation
            if self.add_mfcc:
                n_features_for_rnn += 6  # 6 MFCCs
        else:
            n_features_for_rnn = self.n_features
            if self.add_rms:
                n_features_for_rnn += 1
            if self.add_sf:
                n_features_for_rnn += 1
            if self.add_formants:
                n_features_for_rnn += 2
            if self.add_pitch:
                n_features_for_rnn += 1
            if self.add_mfcc:
                n_features_for_rnn += 6

        # Feature normalization
        self.norm_ = nn.BatchNorm1d(n_features_for_rnn)

        # Feature fusion
        self.fusion_ = nn.Linear(n_features_for_rnn, 64)
        n_features_for_rnn = 64

        if rnn is None:
            rnn = dict()
        self.rnn = rnn
        self.rnn_ = RNN(n_features_for_rnn, **rnn)
        n_features = self.rnn_.dimension

        # Attention mechanism
        self.attention_ = nn.MultiheadAttention(n_features, num_heads=4, dropout=0.1)
        self.attn_norm_ = nn.LayerNorm(n_features)

        if ff is None:
            ff = dict()
        self.ff = ff
        self.ff_ = FF(n_features, **ff)
        n_features = self.ff_.dimension

        if self.task.is_representation_learning:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
            return

        self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
        self.activation_ = self.task.default_activation

    def compute_spectral_flatness(self, output, epsilon=1e-10):
        abs_output = torch.abs(output)
        log_abs_output = torch.log(abs_output + epsilon)
        mean_log = torch.mean(log_abs_output, dim=2)
        exp_mean_log = torch.exp(mean_log)
        mean_abs_output = torch.mean(abs_output, dim=2)
        spectral_flatness = exp_mean_log / (mean_abs_output + epsilon)
        return spectral_flatness.unsqueeze(2)

    def compute_formants(self, waveforms, n_frames, sample_rate=16000):
        formants_list = []
        for waveform_single in waveforms.cpu().numpy(): # Iterate over each waveform in the batch
            try:
                # Ensure waveform_single is 1D for librosa.lpc
                waveform_1d = waveform_single.squeeze()
                if waveform_1d.ndim == 0: # Handle case where squeeze results in 0-dim tensor
                    waveform_1d = np.array([waveform_1d.item()])
                elif waveform_1d.size == 0: # Handle empty waveform
                    raise ValueError("Empty waveform provided to librosa.lpc")

                lpc_order = 2 + sample_rate // 1000
                # Ensure lpc_order is less than the length of the waveform segment
                if lpc_order >= len(waveform_1d):
                    lpc_order = max(1, len(waveform_1d) -1) # Adjust lpc_order if too large

                if len(waveform_1d) == 0 or lpc_order <= 0: # Skip if waveform is too short or lpc_order is invalid
                    freqs = np.zeros(2)
                else:
                    lpc_coeffs = librosa.lpc(waveform_1d, order=lpc_order)
                    roots = np.roots(lpc_coeffs)
                    roots = roots[np.imag(roots) >= 0]
                    angles = np.angle(roots)
                    raw_freqs = angles * (sample_rate / (2 * np.pi))
                    # Filter frequencies to be within a reasonable range and sort
                    valid_freqs = raw_freqs[np.logical_and(raw_freqs > 50, raw_freqs < (sample_rate / 2 - 100))] # Ensure below Nyquist
                    valid_freqs = np.sort(valid_freqs)
                    
                    if len(valid_freqs) >= 2:
                        freqs = valid_freqs[:2]  # Take the first two
                    elif len(valid_freqs) == 1:
                        freqs = np.array([valid_freqs[0], 0.0]) # Pad if only one formant
                    else:
                        freqs = np.zeros(2) # Default if no valid formants found
            except Exception as e:
                # logger.warning(f"Error computing formants: {e}. Returning zeros.") # Optional: log the error
                freqs = np.zeros(2)
            formants_list.append(freqs)
        formants_tensor = torch.tensor(formants_list, dtype=torch.float32, device=waveforms.device)
        # formants_tensor shape: (batch_size, 2)
        return formants_tensor.unsqueeze(1).repeat(1, n_frames, 1) # Repeat across n_frames

    def compute_pitch_variation(self, waveforms, n_frames, sample_rate=16000):
        pitches_list = []
        for waveform_single in waveforms.cpu().numpy(): # Iterate over each waveform in the batch
            try:
                waveform_1d = waveform_single.squeeze()
                if waveform_1d.ndim == 0: waveform_1d = np.array([waveform_1d.item()])
                if waveform_1d.size == 0: raise ValueError("Empty waveform")

                pitch_track, voiced_flag, voiced_probs = librosa.pyin(waveform_1d, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sample_rate)
                valid_pitches = pitch_track[~np.isnan(pitch_track) & (pitch_track > 0)]
                pitch_var_val = np.var(valid_pitches) if len(valid_pitches) > 1 else 0.0
            except Exception as e:
                # logger.warning(f"Error computing pitch variation: {e}. Returning zero.") # Optional: log the error
                pitch_var_val = 0.0
            pitches_list.append(pitch_var_val)
        pitch_var_tensor = torch.tensor(pitches_list, dtype=torch.float32, device=waveforms.device)
        # pitch_var_tensor shape: (batch_size)
        return pitch_var_tensor.unsqueeze(1).unsqueeze(2).repeat(1, n_frames, 1) # Repeat across n_frames

    def compute_mfcc(self, waveforms, n_frames, sample_rate=16000, n_mfcc=6):
        mfccs_list = []
        for waveform_single in waveforms.cpu().numpy(): # Iterate over each waveform in the batch
            try:
                waveform_1d = waveform_single.squeeze()
                if waveform_1d.ndim == 0: waveform_1d = np.array([waveform_1d.item()])
                if waveform_1d.size == 0: raise ValueError("Empty waveform")

                mfcc_features = librosa.feature.mfcc(y=waveform_1d, sr=sample_rate, n_mfcc=n_mfcc)
                mfcc_mean = np.mean(mfcc_features, axis=1)  # Average over time frames of MFCC
            except Exception as e:
                # logger.warning(f"Error computing MFCCs: {e}. Returning zeros.") # Optional: log the error
                mfcc_mean = np.zeros(n_mfcc)
            mfccs_list.append(mfcc_mean)
        mfccs_tensor = torch.tensor(mfccs_list, dtype=torch.float32, device=waveforms.device)
        # mfccs_tensor shape: (batch_size, n_mfcc)
        return mfccs_tensor.unsqueeze(1).repeat(1, n_frames, 1) # Repeat across n_frames

    def forward(self, waveforms, return_intermediate=None):
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms. In case SincNet is skipped, a tensor with shape
            (batch_size, n_samples, n_features) is expected.
        return_intermediate : `int`, optional
            Index of RNN layer. Returns RNN intermediate hidden state.
            Defaults to only return the final output.

        Returns
        -------
        output : `torch.Tensor`
            Final network output.
        intermediate : `torch.Tensor`
            Intermediate network output (only when `return_intermediate`
            is provided).
        """
        if self.sincnet.get('skip', False):
            output = waveforms
        else:
            output = self.sincnet_(waveforms)

        n_frames_target = output.size(1) # Get the number of frames from SincNet's output

        augmented_output = output

        if self.add_rms:
            print("Adding RMS energy feature", flush=True)
            rms = torch.sqrt(torch.mean(waveforms ** 2, dim=1, keepdim=True) + 1e-10)
            rms_expanded = rms.repeat(1, n_frames_target, 1) # Use n_frames_target
            augmented_output = torch.cat((augmented_output, rms_expanded), dim=2)

        if self.add_sf:
            print("Adding spectral flatness feature", flush=True)
            sf = self.compute_spectral_flatness(output)
            augmented_output = torch.cat((augmented_output, sf), dim=2)

        if self.add_formants:
            print("Adding formants feature", flush=True)
            formants = self.compute_formants(waveforms, n_frames=n_frames_target) # Pass n_frames_target
            augmented_output = torch.cat((augmented_output, formants), dim=2)

        if self.add_pitch:
            print("Adding pitch variation feature", flush=True)
            pitch_var = self.compute_pitch_variation(waveforms, n_frames=n_frames_target) # Pass n_frames_target
            augmented_output = torch.cat((augmented_output, pitch_var), dim=2)

        if self.add_mfcc:
            print("Adding MFCC feature", flush=True)
            mfcc = self.compute_mfcc(waveforms, n_frames=n_frames_target) # Pass n_frames_target
            augmented_output = torch.cat((augmented_output, mfcc), dim=2)

        # Normalize features
        batch_size, seq_len, feat_dim = augmented_output.shape
        augmented_output = augmented_output.transpose(1, 2)  # (batch, feat_dim, seq_len)
        augmented_output = self.norm_(augmented_output)  # Normalize
        augmented_output = augmented_output.transpose(1, 2)  # (batch, seq_len, feat_dim)

        # Feature fusion
        augmented_output = self.fusion_(augmented_output)
        augmented_output = torch.relu(augmented_output)

        # RNN
        if return_intermediate is None:
            output = self.rnn_(augmented_output)
        else:
            if return_intermediate == 0:
                intermediate = augmented_output
                output = self.rnn_(augmented_output)
            else:
                output, intermediate = self.rnn_(augmented_output, return_intermediate=True)
                intermediate = intermediate[return_intermediate - 1]

        # Attention
        output = output.transpose(0, 1)  # (seq_len, batch, feat_dim)
        attn_output, _ = self.attention_(output, output, output)
        output = self.attn_norm_(attn_output + output)  # Residual connection
        output = output.transpose(0, 1)  # (batch, seq_len, feat_dim)

        # Feedforward
        output = self.ff_(output)

        if self.task.is_representation_learning:
            output = self.embedding_(output)
            return output if return_intermediate is None else (output, intermediate)

        output = self.linear_(output)
        output = self.activation_(output)
        return output if return_intermediate is None else (output, intermediate)

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension
        return Model.dimension.fget(self)

    def intermediate_dimension(self, layer):
        if layer == 0:
            return self.sincnet_.dimension
        return self.rnn_.intermediate_dimension(layer - 1)


class SincTDNN(Model):
    """waveform -> SincNet -> XVectorNet (TDNN -> FC) -> output

    Parameters
    ----------
    sincnet : `dict`, optional
        SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters.
    tdnn : `dict`, optional
        X-Vector Time-Delay neural network parameters.
        Defaults to `pyannote.audio.models.tdnn.XVectorNet` default parameters.
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
    """

    @staticmethod
    def get_alignment(sincnet=None, **kwargs):
        """
        """

        if sincnet is None:
            sincnet = dict()

        return SincNet.get_alignment(**sincnet)

    supports_packed = False

    @staticmethod
    def get_resolution(sincnet : Optional[dict] = None, **kwargs) -> Resolution:
        """Get sliding window used for feature extraction

        Parameters
        ----------
        sincnet : dict, optional

        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
        """

        # TODO add support for frame-wise and sequence labeling tasks
        # TODO https://github.com/pyannote/pyannote-audio/issues/290
        return RESOLUTION_CHUNK

    def init(self,
             sincnet : Optional[dict] = None,
             tdnn : Optional[dict] = None,
             embedding : Optional[dict] = None):
        """waveform -> SincNet -> XVectorNet (TDNN -> FC) -> output

        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters.
        tdnn : `dict`, optional
            X-Vector Time-Delay neural network parameters.
            Defaults to `pyannote.audio.models.tdnn.XVectorNet` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        n_features = self.n_features

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet

        if n_features != 1:
            raise ValueError('SincNet only supports mono waveforms. '
                             f'Here, waveform has {n_features} channels.')
        self.sincnet_ = SincNet(**sincnet)
        n_features = self.sincnet_.dimension

        if tdnn is None:
            tdnn = dict()
        self.tdnn = tdnn
        self.tdnn_ = XVectorNet(n_features, **tdnn)
        n_features = self.tdnn_.dimension

        if self.task.is_representation_learning:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
        else:
            self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
            self.activation_ = self.task.default_activation

    def forward(self, waveforms: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms

        Returns
        -------
        output : `torch.Tensor`
            Final network output or intermediate network output
            (only when `return_intermediate` is provided).
        """

        output = self.sincnet_(waveforms)

        return_intermediate = 'segment6' if self.task.is_representation_learning else None
        output = self.tdnn_(output, return_intermediate=return_intermediate)

        if self.task.is_representation_learning:
            return self.embedding_(output)

        return self.activation_(self.linear_(output))


    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        return Model.dimension.fget(self)
