global warnings_enabled = true;

function warnings(val)
    global warnings_enabled = val
end

function warn(msg)
    if warnings_enabled
        @warn(msg)
    end
end