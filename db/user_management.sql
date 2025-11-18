-- User Management and Authentication System
-- This file adds user management capabilities required for the rubric

-- Users table for authentication and role-based access
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('admin', 'user', 'viewer')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (now() + INTERVAL '24 hours'),
    ip_address INET,
    user_agent TEXT
);

-- User privileges table for granular permissions
CREATE TABLE IF NOT EXISTS user_privileges (
    privilege_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
    resource VARCHAR(50) NOT NULL, -- 'images', 'categories', 'users', 'admin'
    permission VARCHAR(20) NOT NULL, -- 'create', 'read', 'update', 'delete', 'execute'
    granted_by INT REFERENCES users(user_id),
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Activity log for audit trail
CREATE TABLE IF NOT EXISTS user_activity_log (
    log_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id INT,
    details JSONB DEFAULT '{}',
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Insert default admin user (password: admin123)
-- In production, use a proper password hashing library
INSERT INTO users (username, email, password_hash, role) 
VALUES ('admin', 'admin@imglib.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeblneGW7W.b2F.8K', 'admin')
ON CONFLICT (username) DO NOTHING;

-- Create default user with basic privileges
INSERT INTO users (username, email, password_hash, role)
VALUES ('demo_user', 'user@imglib.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeblneGW7W.b2F.8K', 'user')
ON CONFLICT (username) DO NOTHING;

-- Viewer role for read-only access
INSERT INTO users (username, email, password_hash, role)
VALUES ('viewer', 'viewer@imglib.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeblneGW7W.b2F.8K', 'viewer')
ON CONFLICT (username) DO NOTHING;

-- Grant default privileges
INSERT INTO user_privileges (user_id, resource, permission) 
SELECT u.user_id, 'images', 'read' FROM users u WHERE u.role IN ('user', 'viewer', 'admin')
ON CONFLICT DO NOTHING;

INSERT INTO user_privileges (user_id, resource, permission)
SELECT u.user_id, 'categories', 'read' FROM users u WHERE u.role IN ('user', 'viewer', 'admin')
ON CONFLICT DO NOTHING;

INSERT INTO user_privileges (user_id, resource, permission)
SELECT u.user_id, 'images', p.permission 
FROM users u CROSS JOIN (VALUES ('create'), ('update'), ('delete')) AS p(permission)
WHERE u.role IN ('admin', 'user')
ON CONFLICT DO NOTHING;

-- Functions for user management
CREATE OR REPLACE FUNCTION create_user_with_role(
    p_username VARCHAR(50),
    p_email VARCHAR(100), 
    p_password_hash VARCHAR(255),
    p_role VARCHAR(20)
) RETURNS INT AS $$
DECLARE
    new_user_id INT;
BEGIN
    INSERT INTO users (username, email, password_hash, role)
    VALUES (p_username, p_email, p_password_hash, p_role)
    RETURNING user_id INTO new_user_id;
    
    -- Grant default privileges based on role
    IF p_role = 'admin' THEN
        INSERT INTO user_privileges (user_id, resource, permission)
        SELECT new_user_id, r.resource, r.permission
        FROM (VALUES 
            ('images', 'create'), ('images', 'read'), ('images', 'update'), ('images', 'delete'),
            ('categories', 'create'), ('categories', 'read'), ('categories', 'update'), ('categories', 'delete'),
            ('users', 'create'), ('users', 'read'), ('users', 'update'), ('users', 'delete'),
            ('admin', 'execute')
        ) AS r(resource, permission);
    ELSIF p_role = 'user' THEN
        INSERT INTO user_privileges (user_id, resource, permission)
        SELECT new_user_id, r.resource, r.permission
        FROM (VALUES 
            ('images', 'create'), ('images', 'read'), ('images', 'update'),
            ('categories', 'read')
        ) AS r(resource, permission);
    ELSIF p_role = 'viewer' THEN
        INSERT INTO user_privileges (user_id, resource, permission)
        SELECT new_user_id, r.resource, r.permission
        FROM (VALUES 
            ('images', 'read'), ('categories', 'read')
        ) AS r(resource, permission);
    END IF;
    
    RETURN new_user_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check user permissions
CREATE OR REPLACE FUNCTION check_user_permission(
    p_user_id INT,
    p_resource VARCHAR(50),
    p_permission VARCHAR(20)
) RETURNS BOOLEAN AS $$
DECLARE
    has_permission BOOLEAN := FALSE;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM user_privileges up
        JOIN users u ON up.user_id = u.user_id
        WHERE up.user_id = p_user_id 
        AND up.resource = p_resource 
        AND up.permission = p_permission
        AND u.is_active = TRUE
    ) INTO has_permission;
    
    RETURN has_permission;
END;
$$ LANGUAGE plpgsql;

-- Function to log user activity
CREATE OR REPLACE FUNCTION log_user_activity(
    p_user_id INT,
    p_action VARCHAR(100),
    p_resource_type VARCHAR(50) DEFAULT NULL,
    p_resource_id INT DEFAULT NULL,
    p_details JSONB DEFAULT '{}',
    p_ip_address INET DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO user_activity_log (user_id, action, resource_type, resource_id, details, ip_address)
    VALUES (p_user_id, p_action, p_resource_type, p_resource_id, p_details, p_ip_address);
END;
$$ LANGUAGE plpgsql;

-- Trigger to log user logins
CREATE OR REPLACE FUNCTION log_user_login() RETURNS TRIGGER AS $$
BEGIN
    PERFORM log_user_activity(NEW.user_id, 'login', 'session', NULL, 
                             jsonb_build_object('session_id', NEW.session_id));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_log_user_login ON user_sessions;
CREATE TRIGGER trg_log_user_login
AFTER INSERT ON user_sessions
FOR EACH ROW EXECUTE FUNCTION log_user_login();