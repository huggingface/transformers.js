import { useMemo } from "react";

const Chunk = ({ chunk, currentTime, onClick, ...props }) => {
    const { text, timestamp } = chunk;
    const [start, end] = timestamp;

    const bolded = start <= currentTime && currentTime < end;

    return (
        <span {...props}>
            {text.startsWith(' ') ? " " : ""}
            <span
                onClick={onClick}
                className="text-md text-gray-600 cursor-pointer hover:text-red-600"
                title={timestamp.map(x => x.toFixed(2)).join(' → ')}
                style={{
                    textDecoration: bolded ? 'underline' : 'none',
                    textShadow: bolded ? '0 0 1px #000' : 'none',
                }}
            >{text.trim()}</span>
        </span>
    )
}

const Transcript = ({ transcript, currentTime, setCurrentTime, ...props }) => {


    const jsonTranscript = useMemo(() => {
        return JSON.stringify(transcript, null, 2)
            // post-process the JSON to make it more readable
            .replace(/( {4}"timestamp": )\[\s+(\S+)\s+(\S+)\s+\]/gm, "$1[$2 $3]");
    }, [transcript]);

    const downloadTranscript = () => {
        const blob = new Blob([jsonTranscript], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'transcript.json';
        a.click();
        URL.revokeObjectURL(url);
    }

    return (<>
        <div {...props}>
            {
                transcript.chunks.map((chunk, i) => <Chunk key={i} chunk={chunk} currentTime={currentTime} onClick={e => {
                    setCurrentTime(chunk.timestamp[0]) // Set to start of chunk
                }} />)
            }
        </div>

        <div className="flex justify-center border-t text-sm text-gray-600 max-h-[150px] overflow-y-auto p-2 scrollbar-thin">
            <button
                className="flex items-center border px-2 py-1 rounded-lg bg-green-400 text-white hover:bg-green-500"
                onClick={downloadTranscript}
            >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-6 mr-1">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
                </svg>
                Download transcript
            </button>
        </div>


    </>)
};
export default Transcript;
