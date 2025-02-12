import { Fragment } from 'react';

const COLOURS = [
    'bg-purple-300',
    'bg-green-300',
    'bg-yellow-300',
    'bg-red-300',
    'bg-blue-300',
]

export function Token({ text, position, margin }) {
    const textWithLineBreaks = text.split('\n').map((line, index, array) => (
        <Fragment key={index}>
            {line}
            {index !== array.length - 1 && <br />}
        </Fragment>
    ));
    return (<span
        style={{ marginLeft: margin }}
        className={`leading-5 text-white/[0.87] ${textWithLineBreaks.length === 1 ? 'inline-block ' : ''}${COLOURS[position % COLOURS.length]}`}>
        {textWithLineBreaks}
    </span>)
}