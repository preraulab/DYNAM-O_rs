function masked = mask_spectrogram_rs(spect_2s, stimes_2s, labels_1s, stimes_1s)
%MASK_SPECTROGRAM_RS  Drop-in replacement for maskSpectrogram, calling the
%   Rust implementation. Sets pass-2 spectrogram pixels to zero outside
%   pass-1 regions AND on the 1-pixel inner perimeter of each region
%   (matches MATLAB's `spect_masked(border_inds) = 0`).
%
%   masked = mask_spectrogram_rs(spect_2s, stimes_2s, labels_1s, stimes_1s)
%
%   labels_1s is the (F x T_1) label image returned by extract_tfpeaks_rs
%   on the pass-1 spectrogram. Both spectrograms must share the freq axis.
    masked = double(py.pydynamo.matlab_api.mask_spectrogram_rs( ...
        py.numpy.array(double(spect_2s)), ...
        py.numpy.array(double(stimes_2s(:))), ...
        py.numpy.array(int64(labels_1s)), ...
        py.numpy.array(double(stimes_1s(:)))));
end
