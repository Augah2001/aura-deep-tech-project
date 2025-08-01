import { FC, ReactNode } from 'react';

const MetricCard: FC<{ label: string; value: ReactNode; unit?: string; colorClass?: string; icon?: ReactNode; valueClass?: string }> = ({ label, value, unit, colorClass = 'text-white', icon, valueClass = 'text-xl lg:text-2xl' }) => (
    <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700 h-full flex flex-col">
        <div className="flex items-center justify-center text-sm font-medium text-gray-400 mb-2">
            {icon}
            <span className="ml-2">{label}</span>
        </div>
        <div className="flex-grow flex items-center justify-center overflow-hidden">
             <div className={`${valueClass} font-bold ${colorClass}`}>{value}<span className="text-lg ml-1">{unit}</span></div>
        </div>
    </div>
);

export default MetricCard;
