import { useState } from "react";
import CrossIcon from "./icons/CrossIcon"

export default function ImagePreview({ src, onRemove, ...props }) {
    const [hover, setHover] = useState(false);

    return (
        <div
            {...props}
            onMouseEnter={() => setHover(true)}
            onMouseLeave={() => setHover(false)}
        >
            <CrossIcon onClick={onRemove} className={`absolute top-0 right-0 cursor-pointer dark:fill-gray-400 dark:text-gray-100 fill-gray-200 text-gray-800 ${hover ? '' : 'hidden'}`} />
            <img src={src} alt="Upload preview" className="w-full h-full object-cover rounded-md" />
        </div>)
}
