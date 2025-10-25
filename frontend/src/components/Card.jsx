import clsx from 'clsx';

export default function Card({ children, className, title, action, ...props }) {
  return (
    <div
      className={clsx(
        'bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden',
        'hover:shadow-md transition-shadow duration-200',
        className
      )}
      {...props}
    >
      {title && (
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          {action}
        </div>
      )}
      <div className="p-6">
        {children}
      </div>
    </div>
  );
}
