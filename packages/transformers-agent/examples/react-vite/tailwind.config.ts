import type { Config } from 'tailwindcss';

export default {
    content: ['./index.html', './src/**/*.{ts,tsx}'],
    theme: {
        extend: {
            colors: {
                canvas: '#f4f1e9',
                ink: '#102a43',
                accent: '#d97706',
                pine: '#1f6f50',
            },
            fontFamily: {
                sans: ['"Space Grotesk"', 'ui-sans-serif', 'sans-serif'],
            },
            boxShadow: {
                soft: '0 12px 30px -16px rgba(16, 42, 67, 0.45)',
            },
        },
    },
    plugins: [],
} satisfies Config;
