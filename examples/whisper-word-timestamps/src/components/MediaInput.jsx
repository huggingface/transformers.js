import { useState, forwardRef, useRef, useImperativeHandle, useEffect, useCallback } from 'react';

const EXAMPLE_URL = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/whisper-timestamps-demo.mp4';

const MediaInput = forwardRef(({ onInputChange, onTimeUpdate, ...props }, ref) => {
    // UI states
    const [dragging, setDragging] = useState(false);
    const fileInputRef = useRef(null);

    // Create a reference to the audio and video elements
    const audioElement = useRef(null);
    const videoElement = useRef(null);

    const currentTimeRef = useRef(0);
    useImperativeHandle(ref, () => ({
        setMediaTime(time) {
            if (audioElement.current?.src) {
                audioElement.current.currentTime = time;
            } else if (videoElement.current?.src) {
                videoElement.current.currentTime = time;
            }
            currentTimeRef.current = time;
        }
    }));

    const onBufferLoad = (arrayBuffer, type) => {
        const blob = new Blob([arrayBuffer.slice(0)], { type: type });
        const url = URL.createObjectURL(blob);
        processFile(arrayBuffer);

        // Create a URL for the Blob
        if (type.startsWith('audio/')) {
            // Dispose the previous source
            videoElement.current.pause();
            videoElement.current.removeAttribute('src');
            videoElement.current.load();

            audioElement.current.src = url;
        } else if (type.startsWith('video/')) {
            // Dispose the previous source
            audioElement.current.pause();
            audioElement.current.removeAttribute('src');
            audioElement.current.load();

            videoElement.current.src = url;
        } else {
            alert(`Unsupported file type: ${type}`);
        }
    }

    const readFile = (file) => {
        if (!file) return;

        // file.type
        const reader = new FileReader();
        reader.onload = (e) => {
            onBufferLoad(e.target.result, file.type);
        }
        reader.readAsArrayBuffer(file);
    }

    const handleInputChange = (event) => {
        readFile(event.target.files[0]);
    };

    const handleDragOver = (event) => {
        event.preventDefault();
    };

    const handleDrop = (event) => {
        event.preventDefault();
        setDragging(false);
        readFile(event.dataTransfer.files[0]);
    };

    const handleClick = (e) => {
        if (e.target.tagName === 'VIDEO' || e.target.tagName === 'AUDIO') {
            e.preventDefault();
            fileInputRef.current.click();
        } else if (e.target.tagName === 'INPUT') {
            e.stopPropagation();
        } else {
            fileInputRef.current.click();
            e.stopPropagation();
        }
    };

    const processFile = async (buffer) => {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16_000 });

        try {
            const audioBuffer = await audioContext.decodeAudioData(buffer);
            let audio;
            if (audioBuffer.numberOfChannels === 2) {
                // Merge channels
                const SCALING_FACTOR = Math.sqrt(2);
                const left = audioBuffer.getChannelData(0);
                const right = audioBuffer.getChannelData(1);
                audio = new Float32Array(left.length);
                for (let i = 0; i < audioBuffer.length; ++i) {
                    audio[i] = SCALING_FACTOR * (left[i] + right[i]) / 2;
                }
            } else {
                audio = audioBuffer.getChannelData(0);
            }
            onInputChange(audio);

        } catch (e) {
            alert(e);
        }
    };

    const requestRef = useRef();

    const updateTime = useCallback(() => {
        let elem;
        if (audioElement.current?.src) {
            elem = audioElement.current;

        } else if (videoElement.current?.src) {
            elem = videoElement.current;
        }

        if (elem && currentTimeRef.current !== elem.currentTime) {
            currentTimeRef.current = elem.currentTime;
            onTimeUpdate(elem.currentTime);
        }

        // Request the next frame
        requestRef.current = requestAnimationFrame(updateTime);
    }, [onTimeUpdate]);

    useEffect(() => {
        // Start the animation
        requestRef.current = requestAnimationFrame(updateTime);

        return () => {
            // Cleanup on component unmount
            cancelAnimationFrame(requestRef.current);
        };
    }, [updateTime]);
    return (
        <div
            {...props}
            onClick={handleClick}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onDragEnter={(e) => setDragging(true)}
            onDragLeave={(e) => setDragging(false)}
        >
            <input
                type="file"
                accept="audio/*,video/*"
                onChange={handleInputChange}
                ref={fileInputRef}
                className="hidden"
            />
            {
                <audio
                    ref={audioElement}
                    controls
                    style={{ display: audioElement.current?.src ? 'block' : 'none' }}
                    className='w-full max-h-full'
                />
            }
            {
                <video
                    ref={videoElement}
                    controls
                    style={{ display: videoElement.current?.src ? 'block' : 'none' }}
                    className='w-full max-h-full'
                />
            }
            {
                !audioElement.current?.src && !videoElement.current?.src && (
                    <div className="w-full flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-md h-[250px]"
                        style={{ borderColor: dragging ? 'blue' : 'lightgray' }}
                    >
                        <span className="text-gray-600 text-center"><u>Drag & drop</u> or <u>click</u><br />to select media</span>
                        <span className="text-gray-500 text-sm hover:text-gray-800 mt-2" onClick={async (e) => {
                            e.stopPropagation();
                            const buffer = await fetch(EXAMPLE_URL).then((r) => r.arrayBuffer());
                            videoElement.current.src = URL.createObjectURL(new Blob([buffer], { type: 'video/mp4' }));
                            onBufferLoad(buffer, 'video/mp4');
                        }}>(or <u>try an example</u>)</span>
                    </div>
                )
            }
        </div>
    );
});
MediaInput.displayName = 'MediaInput';

export default MediaInput;
