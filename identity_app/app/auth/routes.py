from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_user, logout_user, current_user, login_required
from app.auth import bp
from app.auth.forms import LoginForm, RegistrationForm
from app.models.user import User
from app import db

@bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login route"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        # Try to find user by username or email
        user = User.query.filter(
            (User.username == form.username_or_email.data) | 
            (User.email == form.username_or_email.data)
        ).first()
        
        if user and user.check_password(form.password.data):
            if not user.is_active:
                flash('Account is deactivated. Please contact support.', 'error')
                return redirect(url_for('auth.login'))
            
            login_user(user, remember=form.remember_me.data)
            user.update_last_login()
            
            flash(f'Welcome back, {user.get_full_name()}!', 'success')
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            if not next_page or not next_page.startswith('/'):
                next_page = url_for('index')
            return redirect(next_page)
        else:
            flash('Invalid username/email or password', 'error')
    
    return render_template('auth/login.html', title='Sign In', form=form)

@bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            user = User(
                username=form.username.data,
                email=form.email.data.lower(),
                first_name=form.first_name.data,
                last_name=form.last_name.data,
                phone_number=form.phone_number.data or None
            )
            user.set_password(form.password.data)
            
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
            
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'error')
            print(f"Registration error: {e}")
    
    return render_template('auth/register.html', title='Register', form=form)

@bp.route('/logout')
@login_required
def logout():
    """User logout route"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@bp.route('/profile')
@login_required
def profile():
    """User profile page"""
    from app.models.verification import Verification
    
    # Get user's recent verification attempts
    recent_verifications = Verification.get_recent_attempts(current_user.id)
    
    # Get success rates
    overall_success_rate = Verification.get_user_success_rate(current_user.id)
    liveness_success_rate = Verification.get_user_success_rate(current_user.id, 'liveness')
    
    return render_template('auth/profile.html',
                         title='Profile',
                         user=current_user,
                         recent_verifications=recent_verifications,
                         overall_success_rate=overall_success_rate,
                         liveness_success_rate=liveness_success_rate)

@bp.route('/api/user-info')
@login_required
def user_info():
    """API endpoint to get current user info"""
    return jsonify({
        'success': True,
        'user': current_user.to_dict()
    })
